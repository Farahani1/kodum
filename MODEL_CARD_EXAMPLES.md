# Kodoom: context, question, and answer examples

[Read the model card](MODEL_CARD.md) for measured results, training information,
and evaluation scope.

These **original, illustrative examples** show the seven data groups discussed
in the model card. They are written for this guide, not copied from the private
dataset or selected benchmark cases. The English and Persian versions describe
the same decision. **Expected answers are explanations of these examples, not
measured Kodoom predictions, dataset gold labels, or new evaluation results.**

In actual use, Kodoom scores the supplied options and returns probabilities.
This guide shows the expected selected answer without inventing model scores.
The measured comparison represents yes/no and score questions as choices too;
the examples preserve that distinction.

| Group in the model card | Example decision | Original question type |
| --- | --- | --- |
| Invoice processing | Route an invoice whose total disagrees with its line items. | Choice |
| Security incidents | Assign an incident level under an explicit severity policy. | Score, represented as choice |
| Agent trace observability | Stop a trace that attempts an operation forbidden by policy. | Choice |
| Customer service | Check refund eligibility against a stated rule. | Yes/no, represented as choice |
| ParsiNLU paraphrase | Decide whether two questions ask the same thing. | Choice |
| PersianQA answerability | Decide whether the passage contains the requested information. | Two-option choice |
| Translated Helmo diagnostic | Select the component implicated by a short service-health log. | Choice |

## 1. Invoice processing: route a discrepancy

**Context (English):** An invoice lists 3 items at 500,000 rials each. There is
no tax, shipping charge, or discount. Its stated total is 1,800,000 rials. The
processing policy says to send an invoice for review when its stated total does
not equal the sum of its line items.

**Context (Persian):** یک فاکتور شامل ۳ کالا با قیمت هر کالا ۵۰۰٬۰۰۰ ریال است.
مالیات، هزینه ارسال و تخفیف ندارد. مبلغ کل درج شده در فاکتور ۱٬۸۰۰٬۰۰۰ ریال
است. طبق دستورالعمل، اگر مبلغ کل با مجموع اقلام برابر نباشد، فاکتور باید
برای بررسی ارجاع شود.

**Question (English):** What should happen to this invoice?

**Question (Persian):** این فاکتور باید به کدام مرحله ارجاع شود؟

| Option ID | English answer | Persian answer |
| --- | --- | --- |
| `approve` | Approve it for payment. | برای پرداخت تایید شود. |
| `review` | Send it for discrepancy review. | برای بررسی مغایرت ارجاع شود. |
| `archive` | Archive it without further action. | بدون اقدام بیشتر بایگانی شود. |

**Expected answer:** `review` — **ارجاع برای بررسی مغایرت**.
The line items total 1,500,000 rials, which differs from the stated total.

This illustrates choosing an action from structured financial context. It does
not imply that Kodoom reads invoice images, performs OCR, or validates actual
payments. The corresponding model-card result comes from sampled translated
typed-decisions test cases.

## 2. Security incidents: choose a severity level

**Context (English):** The incident policy defines three levels: low for a
blocked attempt with no successful access; medium for suspicious access without
confirmed privileged changes; high for confirmed privileged access followed by
an unauthorized configuration change. This incident contains a confirmed
administrator login and an unauthorized change to a production configuration.

**Context (Persian):** دستورالعمل رخداد سه سطح دارد: سطح پایین برای تلاش
مسدود شده بدون دسترسی موفق؛ سطح متوسط برای دسترسی مشکوک بدون تغییر تایید شده
با دسترسی ویژه؛ و سطح بالا برای دسترسی ویژه تایید شده که پس از آن تغییر
غیرمجاز در تنظیمات رخ داده باشد. در این رخداد، ورود مدیر سامانه و تغییر
غیرمجاز در تنظیمات محیط عملیاتی تایید شده است.

**Question (English):** Which incident level applies under this policy?

**Question (Persian):** طبق این دستورالعمل، سطح رخداد کدام است؟

| Option ID | English answer | Persian answer |
| --- | --- | --- |
| `low` | Low | پایین |
| `medium` | Medium | متوسط |
| `high` | High | بالا |

**Expected answer:** `high` — **بالا**.
The context explicitly satisfies the stated high-severity rule.

This illustrates a score question represented as a choice over named levels.
It measures selection under the supplied policy, rather than an independently
validated assessment of real security risk. The model-card comparison uses
the same score-to-choice approach for all three models.

## 3. Agent trace observability: stop a forbidden operation

**Context (English):** An agent may inspect the project workspace but must not
delete anything outside it. Its current trace proposes a delete operation on
an unrelated directory outside the workspace. The oversight policy says to
stop execution before an explicitly forbidden operation is performed.

**Context (Persian):** یک عامل اجازه دارد پوشه پروژه را بررسی کند، اما نباید
هیچ چیزی را بیرون از آن حذف کند. در ردگیری فعلی، عامل پیشنهاد داده است یک
پوشه نامرتبط بیرون از پوشه پروژه را حذف کند. طبق دستورالعمل نظارت، اجرای
عامل باید پیش از انجام یک عملیات صریحا ممنوع متوقف شود.

**Question (English):** What should the oversight system do with this trace?

**Question (Persian):** سامانه نظارت باید با این ردگیری چه کند؟

| Option ID | English answer | Persian answer |
| --- | --- | --- |
| `continue` | Let the agent continue. | اجازه دهد عامل ادامه دهد. |
| `observe` | Continue and flag the trace for later inspection. | اجرا ادامه یابد و ردگیری برای بررسی بعدی علامت بخورد. |
| `human_review` | Queue the trace for human review. | ردگیری در صف بررسی انسانی قرار گیرد. |
| `stop` | Stop the agent before the proposed operation. | عامل پیش از عملیات پیشنهادی متوقف شود. |

**Expected answer:** `stop` — **توقف عامل پیش از عملیات**.
The operation violates an explicit boundary, and the supplied policy specifies
the response.

This illustrates a decision about an agent trace. The document describes the
decision interface; Kodoom itself does not execute or intercept the operation.

## 4. Customer service: check a refund condition

**Context (English):** The store permits a refund only if the customer requests
it within 14 days of delivery and the item is unopened. The customer received
the item 8 days ago and confirms that it is still unopened.

**Context (Persian):** فروشگاه فقط زمانی وجه را پس می دهد که مشتری حداکثر
تا ۱۴ روز پس از تحویل درخواست بدهد و بسته کالا باز نشده باشد. مشتری کالا را
۸ روز پیش دریافت کرده و تایید می کند که بسته هنوز باز نشده است.

**Question (English):** Does this request meet the stated refund conditions?

**Question (Persian):** آیا این درخواست شرایط اعلام شده برای استرداد وجه را دارد؟

| Option ID | English answer | Persian answer |
| --- | --- | --- |
| `A` | Yes, both conditions are met. | بله، هر دو شرط برقرار است. |
| `B` | No, at least one condition is not met. | خیر، دست کم یکی از شرط ها برقرار نیست. |

**Expected answer:** `A` — **بله**.
Eight days is within the 14-day window, and the item is unopened.

This illustrates the yes/no (`noul`) decision type represented as a two-option
choice. The policy is part of the example context; the model need not invent
the store's rules.

## 5. ParsiNLU paraphrase: compare question meaning

**Context (English):** Question 1: "How can I change the password for my
account?" Question 2: "What steps let me set a new password for my account?"

**Context (Persian):** پرسش اول: «چطور رمز عبور حسابم را تغییر بدهم؟»
پرسش دوم: «برای تعیین رمز عبور جدید برای حسابم چه مراحلی لازم است؟»

**Question (English):** Do the two questions ask for the same information?

**Question (Persian):** آیا این دو پرسش اطلاعات یکسانی را درخواست می کنند؟

| Option ID | English answer | Persian answer |
| --- | --- | --- |
| `same` | They ask for the same information. | اطلاعات یکسانی را درخواست می کنند. |
| `different` | They ask for different information. | اطلاعات متفاوتی را درخواست می کنند. |

**Expected answer:** `same` — **معنای یکسان**.
Both questions ask how to change the account password.

This illustrates the task format of the ParsiNLU paraphrase sample, not an
actual ParsiNLU record. It is a decision about semantic equivalence, not a
request to generate a paraphrase. The model card reports a small observed
accuracy gain on this task, with uncertainty intervals that include zero.

## 6. PersianQA answerability: check whether evidence is present

**Context (English):** Passage: "The museum opens at 9 a.m. and closes at
5 p.m. every day except Monday. It has separate galleries for painting and
photography." Reader question: "How much does an admission ticket cost?"

**Context (Persian):** متن: «موزه هر روز به جز دوشنبه از ساعت ۹ صبح تا
۵ عصر باز است. موزه نگارخانه های جداگانه ای برای نقاشی و عکاسی دارد.»
پرسش خواننده: «قیمت بلیت ورودی چقدر است؟»

**Decision question (English):** Does this passage contain the answer to the
reader's question?

**Decision question (Persian):** آیا پاسخ پرسش خواننده در این متن وجود دارد؟

| Option ID | English answer | Persian answer |
| --- | --- | --- |
| `A` | Yes, the answer is in the passage. | بله، پاسخ در متن وجود دارد. |
| `B` | No, the answer is absent from the passage. | خیر، پاسخ در متن وجود ندارد. |

**Expected answer:** `B` — **خیر، پاسخ در متن نیست**.
The passage gives hours and gallery information, but no admission price.

This illustrates the answerability conversion used for PersianQA. The decision
is whether the evidence contains an answer; it does not ask Kodoom to produce
the ticket price. The model-card accuracy is for that two-option decision,
not extracted or generated answers. This is an original example, not a
PersianQA passage.

## 7. Translated Helmo diagnostic: use a technical log

**Context (English):** A service-health log records the following checks. The
runbook says to route the alert to the owner of the only component whose check
fails.

**Context (Persian):** گزارش سلامت سرویس، نتیجه بررسی های زیر را ثبت کرده
است. طبق راهنمای عملیات، هشدار باید به مسئول تنها مولفه ای ارجاع شود که
بررسی آن ناموفق بوده است.

```text
api_health=ok
cache_health=timeout
queue_health=ok
```

**Question (English):** Which component owner should receive this alert?

**Question (Persian):** این هشدار باید به مسئول کدام مولفه ارجاع شود؟

| Option ID | English answer | Persian answer |
| --- | --- | --- |
| `api` | API owner | مسئول API |
| `cache` | Cache owner | مسئول حافظه نهان |
| `queue` | Queue owner | مسئول صف |

**Expected answer:** `cache` — **مسئول حافظه نهان**.
Only the cache check fails. The machine-readable log stays identical between
the English and Persian contexts.

This illustrates one technical scenario of the kind covered by the broader
Helmo diagnostic. That diagnostic spans many topics; a single log example does
not represent every topic or establish a topic-specific performance gain.
The model card keeps Helmo's source-`train` diagnostic separate from held-out
benchmark results. This example is not a Helmo record.

## Reading the answers correctly

These examples make the interface concrete: **context + question + candidate
answers -> probabilities over those answers**. A production integration can
select the top option or send uncertain cases for review using thresholds
validated on its own data.

The dataset's actual workflow targets can distribute probability across several
plausible answers. The unambiguous expected answers in this guide are teaching
examples, not a replacement for those soft targets. No performance claims or
English benchmark scores are derived from this document.
