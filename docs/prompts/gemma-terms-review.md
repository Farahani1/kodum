# Prompt: do the Gemma terms allow an Apache-2.0 typed-decisions-fa?

Give the prompt below to an AI model that can browse the web (it must read the current
terms, not recall them). Plan 1.2 says to read the Gemma terms on generated outputs before
publishing any translation; the answer goes into the plan and the dataset card. It is not
legal advice: if the answer is unclear or says "it depends", ask a lawyer before release.

---

You are reviewing licence terms for an open-source dataset project. Read the current
documents yourself from the official sources and quote them; do not answer from memory, and
say so if a page cannot be reached.

## The project

kodoom builds **typed-decisions-fa**, a Persian translation of the English dataset
`LocalLLaMA/typed-decisions` on Hugging Face (licence **Apache-2.0**). Each record is a short
business case (customer service, invoices, security incidents, AI-agent logs) with a
question, answer options and probability labels. Only the natural-language text is
translated; the labels, IDs and numbers are copied unchanged from the English source. People
review the translations and fix errors by hand.

The translations are produced by Google models run locally on a Colab GPU, downloaded from
Hugging Face after accepting the Gemma licence:

- `google/translategemma-4b-it` (TranslateGemma)
- `google/gemma-3-4b-it` and `google/gemma-3-12b-it` (instruction-tuned Gemma 3, used as
  translators with a glossary in the prompt)

Planned uses of the output:

1. Publish typed-decisions-fa on Hugging Face under **Apache-2.0**, with attribution to the
   English source and a statement of which model translated it.
2. Use it, with other data, to fine-tune open models: `jhu-clsp/mmBERT-base` (an encoder)
   and `Qwen/Qwen3.5-0.8B` (a decoder), and publish those models under a permissive licence
   that is not decided yet.
3. Use it as a public benchmark that anyone may run, including commercially.

## Read these documents (current versions)

- The Gemma Terms of Use (ai.google.dev/gemma/terms)
- The Gemma Prohibited Use Policy (ai.google.dev/gemma/prohibited_use_policy)
- The Hugging Face model pages and licence notices of the three models above, and any
  TranslateGemma-specific terms, model card or paper section about licensing
- Anything these documents incorporate by reference

Record the URL and the "last modified" date (or the date you accessed it) of each.

## Answer these questions

For each, give a short answer (**yes**, **no**, or **unclear**), the exact sentences that
decide it, with their section numbers, and your reasoning in two or three sentences.

1. **Ownership of outputs.** Do the terms claim any rights in text the models generate? Who
   owns the translations?
2. **Outputs as a "Model Derivative".** How do the terms define "Model Derivatives" (or the
   current equivalent)? Is a dataset of model outputs itself a Model Derivative, or only a
   model trained on such outputs (for example through distillation or synthetic data)?
3. **Licensing the dataset.** Can typed-decisions-fa be published under plain Apache-2.0? If
   the terms impose anything on the outputs (use restrictions, a notice, passing on the
   Prohibited Use Policy, a copy of the terms), list each obligation and whether it is
   compatible with Apache-2.0 as the dataset's only licence.
4. **Training other models on the dataset.** If mmBERT or Qwen is fine-tuned on data that
   includes these translations, does the fine-tuned model become a Gemma Model Derivative?
   If so, what must its release carry (the Gemma terms, use restrictions, notices), and can
   it still be released under Apache-2.0 or MIT? Does the answer change if the Persian text
   was reviewed and corrected by people, or if the translations are only a part of the
   training data?
5. **Third-party use.** Would people who download typed-decisions-fa, or the models trained
   on it, be bound by the Gemma terms or the Prohibited Use Policy?
6. **Attribution.** Must the dataset card or model card name Gemma, include a notice, or link
   the terms? Write the exact wording to use, if any is required.
7. **TranslateGemma specifically.** Is TranslateGemma under the same Gemma terms, or does it
   have its own licence with different rules on outputs?
8. **Changes over time.** Can Google change the terms in a way that would bind outputs made
   earlier? Quote the clause about updates.

## Then give

- A **verdict** in one paragraph: can the project keep its plan (dataset under Apache-2.0,
  models under a permissive licence) if it translates with these models? Choose one of:
  *allowed as planned*, *allowed with conditions* (list them), or *not allowed*.
- A **table**: planned use | allowed? | condition | deciding clause.
- **Options if the answer is not "allowed as planned"**: for example a dataset licence that
  passes on Gemma's terms, translating the published test split with a different open model
  under a permissive licence, or keeping Gemma output out of the training mix, with what
  each option costs the project.
- **What stays uncertain**, and the specific questions to put to a lawyer.

Be precise and conservative: when the text is ambiguous, say so and quote it rather than
picking the convenient reading.
