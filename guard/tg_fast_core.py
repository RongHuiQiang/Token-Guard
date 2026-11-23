#!/usr/bin/env python3
import sys
# -*- coding: utf-8 -*-
"""
tg_fast_core.py — Token-Guard 官方管线的通用加速组件（骨干无关）。

从 tg_qwen_runner_fast.py 抽出的四个通用件，只依赖 torch/transformers 和
官方 token_guard_plugin / generation_utils：
  AllSeqsStopString  逐序列停在自己的第一个 stop 串（全 batch 停才停），
                     替代"生成满 max_new_tokens 再事后字符串截断"
  _row_logprobs      增量逐行 logprob，替代 compute_transition_scores 的
                     [B, steps, V] 单张量 stack（B=40 时 23.2GiB OOM 源头）
  FastGenerator      批量+真停止+空文本→-inf+completion_cap 的生成器
  FastLatentEnvironment  verify 批处理（ctx+候选右 padding、自适应子批、
                     fp16 同算术路径）+ 论文 Eq.10 的 3-token 窗口 refine

默认行为开关：completion_cap（默认 None=不 cap，官方行为）；refine_mode
（默认 'suffix'=官方代码行为，'window'=论文 Eq.10 口径）。
"""
import os
import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import token_guard_plugin as _tgp
from generation_utils import TokenGuardGenerator


class AllSeqsStopString(StoppingCriteria):
    """Mark a sequence done once any stop string appears in ITS generated text;
    halt the whole batch only when every sequence is done."""

    def __init__(self, tokenizer, stop_strs, batch_size, gen_start):
        self.tokenizer  = tokenizer
        self.stop_strs  = [s for s in (stop_strs or []) if s]
        self.done       = torch.zeros(batch_size, dtype=torch.bool)
        self.done_step  = [None] * batch_size   # generated-token count at hit
        self.gen_start  = gen_start             # padded prompt length

    @torch.no_grad()
    def __call__(self, input_ids, scores, **kwargs):
        if not self.stop_strs:
            return False
        L = input_ids.shape[1]
        for i in range(input_ids.shape[0]):
            if self.done[i] or L <= self.gen_start:
                continue
            txt = self.tokenizer.decode(
                input_ids[i, self.gen_start:], skip_special_tokens=True
            )
            if any(s in txt for s in self.stop_strs):
                self.done[i]    = True
                self.done_step[i] = L - self.gen_start - 1
        return bool(self.done.all())


def _row_logprobs(sequences, scores, input_len, device):
    """Per-token log-probs [B, steps] computed incrementally.

    Same math as model.compute_transition_scores(..., normalize_logits=True)
    but WITHOUT stacking the per-step score tensors into a [B, steps, V]
    monster (that single fp32 stack reached 23.2 GiB at B=40)."""
    lps = []
    for s, sc in enumerate(scores):
        sc = sc.float()
        lse = torch.logsumexp(sc, dim=-1)
        pos = input_len + s
        if pos >= sequences.shape[1]:
            break
        tok = sequences[:, pos]
        lp = sc.gather(1, tok.unsqueeze(1)).squeeze(1) - lse
        lps.append(lp)
    if not lps:
        return torch.zeros((sequences.shape[0], 1), device=device)
    return torch.stack(lps, dim=1)


class FastGenerator(TokenGuardGenerator):
    """Same sampling params as the official generator; batched with real
    per-sequence stopping and bounded memory."""

    MAX_ROWS_PER_CALL = 16   # bounds the output_scores tuple ≈ 10GB
    completion_cap = None    # set to step_beam_size to cap completions

    def generate(self, prompts, n_return, max_new_tokens=1024, stop_strs=None):
        responses, cum_logprobs = [], []
        if (self.completion_cap is not None and n_return == 1
                and max_new_tokens == 1024 and len(prompts) > self.completion_cap):
            prompts = prompts[:self.completion_cap]
        per_call_prompts = max(1, self.MAX_ROWS_PER_CALL // max(n_return, 1))
        for c0 in range(0, len(prompts), per_call_prompts):
            r, lp = self._generate_chunk(
                prompts[c0: c0 + per_call_prompts], n_return,
                max_new_tokens, stop_strs)
            responses.extend(r)
            cum_logprobs.extend(lp)
        while len(responses) < len(prompts) * n_return:
            responses.append("")
            cum_logprobs.append(-1.0)
        return responses, cum_logprobs

    def _generate_chunk(self, prompts, n_return, max_new_tokens, stop_strs):
        responses, cum_logprobs = [], []

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=4096,
            )
        finally:
            self.tokenizer.padding_side = old_side

        input_ids      = enc.input_ids.to(self.model.device)
        attention_mask = enc.attention_mask.to(self.model.device)
        input_len      = input_ids.shape[1]

        stop_crit = AllSeqsStopString(
            self.tokenizer, stop_strs, input_ids.shape[0] * n_return, input_len
        )
        stopping = StoppingCriteriaList([stop_crit]) if stop_strs else None

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if stopping is not None:
            gen_kwargs["stopping_criteria"] = stopping

        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    do_sample=True, temperature=0.7,
                    top_p=0.95, top_k=50,
                    repetition_penalty=1.0,
                    num_return_sequences=n_return,
                    **gen_kwargs,
                )
            except RuntimeError as e:
                print(f"[TG-fast] fallback greedy: {e}", flush=True)
                outputs = self.model.generate(
                    do_sample=False, num_return_sequences=1,
                    **gen_kwargs,
                )
                if n_return > 1:
                    outputs.sequences = outputs.sequences.repeat(n_return, 1)
                    if outputs.scores:
                        outputs.scores = tuple(
                            s.repeat(n_return, 1) for s in outputs.scores
                        )

        if outputs.scores:
            ts = _row_logprobs(outputs.sequences, outputs.scores, input_len,
                               self.model.device)
        else:
            ts = torch.zeros((n_return, 1), device=self.model.device)

        for i in range(outputs.sequences.shape[0]):
            gen_ids  = outputs.sequences[i][input_len:]
            text     = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            kept_len = len(gen_ids)
            if stop_strs:
                for s in stop_strs:
                    if s in text:
                        text = text.split(s)[0]
                        ds = stop_crit.done_step[i]
                        if ds is not None:
                            kept_len = min(kept_len, ds + 1)
                        break
            responses.append(text)
            # empty kept text → -inf: matches the official engine's effective
            # pruning statistics (empty rollouts averaged to -inf over the
            # discarded tail) and caps completions per step.
            if not text.strip():
                cum_logprobs.append(float('-inf'))
                continue
            valid = min(kept_len, ts.shape[1]) if ts.numel() > 0 else 0
            avg_lp = (ts[i, :valid].sum().item() / (valid + 1e-8)) if valid > 0 else -1.0
            cum_logprobs.append(avg_lp)

        return responses, cum_logprobs


class FastLatentEnvironment(_tgp.LatentEnvironment):
    """verify_candidates batched (right padding, adaptive sub-batching, fp16
    arithmetic path identical to official); optional paper-Eq.10 window
    refine. refine_mode: 'suffix' = official behavior (default),
    'window' = 3-token window per paper."""

    refine_mode = 'suffix'

    def verify_candidates(self, context_text, candidate_texts, h_x):
        if not candidate_texts:
            return []
        if h_x.device.type != str(self.device):
            h_x = h_x.to(self.device)

        ctx_inputs = self.tokenizer(
            context_text, return_tensors="pt",
            truncation=True, max_length=8192,
        )
        ctx_len = ctx_inputs.input_ids.shape[1]

        texts = [t for t in candidate_texts if t.strip()]
        if not texts:
            return []

        full_texts = [context_text + t for t in texts]
        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            enc = self.tokenizer(
                full_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=8192,
            )
        finally:
            self.tokenizer.padding_side = old_side

        all_ids  = enc.input_ids
        all_mask = enc.attention_mask
        L_max    = all_ids.shape[1]

        # adaptive sub-batching: cap per-forward footprint
        # ([B, L, V] logits + (n_layers+1)×[B, L, d] hidden tensors)
        vocab = getattr(self.model.config, "vocab_size", 128256)
        d_model = getattr(self.model.config, "hidden_size", 4096)
        n_layers = getattr(self.model.config, "num_hidden_layers", 32)
        per_cand_bytes = L_max * (2 * vocab + 2 * d_model * (n_layers + 2))
        try:
            free_b, _total = torch.cuda.mem_get_info(self.device)
            budget = max(int(free_b * 0.35), 1 << 30)
        except Exception:
            budget = 8 << 30
        bs = max(1, budget // max(per_cand_bytes, 1))

        results = []
        for start in range(0, len(texts), bs):
            chunk_texts = texts[start: start + bs]
            input_ids      = all_ids[start: start + bs].to(self.device)
            attention_mask = all_mask[start: start + bs].to(self.device)
            seq_lens       = attention_mask.sum(dim=1)

            with torch.no_grad():
                outputs = self.model(
                    input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False,
                )
            hiddens = outputs.hidden_states[-2]
            embeds  = outputs.hidden_states[0]
            logits  = outputs.logits

            for row, text in enumerate(chunk_texts):
                n = int(seq_lens[row])
                current_ctx_len = min(ctx_len, n - 1)
                if current_ctx_len >= n:
                    continue

                seg_logits = logits[row, current_ctx_len - 1: n - 1, :]
                seg_ids    = input_ids[row, current_ctx_len: n]
                seg_h      = hiddens[row, current_ctx_len: n, :]
                seg_e      = embeds[row, current_ctx_len: n, :]

                log_probs  = torch.nn.functional.log_softmax(seg_logits, dim=-1)
                tok_lp     = log_probs.gather(1, seg_ids.unsqueeze(1)).squeeze(1)
                tok_p      = tok_lp.exp()

                token_scores = []
                running = h_x.clone()
                t_offset = 1
                for i in range(seg_h.shape[0]):
                    h_t  = seg_h[i]
                    sim  = torch.nn.functional.cosine_similarity(
                        h_t, running, dim=0).float()
                    p_t  = tok_p[i].float()
                    f_token = (self.config.lambda_val * sim
                               + (1 - self.config.lambda_val) * p_t)
                    token_scores.append(f_token)
                    if f_token >= self.config.tau_token:
                        running = (running * t_offset + h_t) / (t_offset + 1)
                        t_offset += 1

                token_scores_t = torch.stack(token_scores).float()

                score_token, H_k = self.calculate_weighted_token_score(
                    token_scores_t, seg_h)
                e_tilde_k = torch.mean(seg_e, dim=0)
                f_norm    = torch.norm(token_scores_t).item()
                score_const = self.calculate_local_consistency(seg_h)
                score_align = self.calculate_global_alignment(H_k, h_x)
                f_seg = self.segment_level_hallucination_score(
                    score_token, score_const, score_align)

                results.append(_tgp.CandidateSegment(
                    text=text,
                    H_k=H_k.cpu(),
                    e_tilde_k=e_tilde_k.cpu(),
                    f_norm=f_norm,
                    segment_score=f_seg,
                    token_scores=[float(x) for x in token_scores_t.cpu().tolist()],
                ))
            del outputs, hiddens, embeds, logits
        return results

    def refine_segment(self, context_text, artifact, h_x, max_retries=3):
        """window mode follows paper Eq.10: rewrite only the 3-token window
        around the weakest token; suffix mode = official whole-suffix regen."""
        if self.refine_mode != 'window':
            return super().refine_segment(context_text, artifact, h_x,
                                          max_retries=max_retries)

        advice = self.check_refinement(artifact)
        if not advice.needed or max_retries <= 0:
            return artifact

        best_artifact = artifact
        for _ in range(max_retries):
            scores = best_artifact.token_scores
            if not scores:
                break
            min_idx = int(np.argmin(scores))
            seg_ids = self.tokenizer.encode(best_artifact.text,
                                            add_special_tokens=False)
            if len(seg_ids) < 4:
                break
            s = max(0, min_idx - 1)
            e = min(len(seg_ids), min_idx + 2)
            if s >= e:
                break

            prefix_text = self.tokenizer.decode(seg_ids[:s],
                                                skip_special_tokens=True)
            inputs = self.tokenizer(
                context_text + prefix_text,
                return_tensors="pt", truncation=True, max_length=4096,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=e - s,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = outputs[0][inputs.input_ids.shape[1]:]
            new_seg_ids = (list(seg_ids[:s]) + gen_ids[:e - s].tolist()
                           + list(seg_ids[e:]))
            new_text = self.tokenizer.decode(new_seg_ids,
                                             skip_special_tokens=True)
            if not new_text.strip():
                break

            refined = self.verify_candidates(context_text, [new_text], h_x)
            if not refined:
                break
            if refined[0].segment_score > best_artifact.segment_score:
                best_artifact = refined[0]
                adj_high = (self.config.tau_seg_high
                            + self.current_threshold_adj)
                if best_artifact.segment_score >= adj_high:
                    break

        return best_artifact
