# MFS Guard: a four-minute demo that earns attention

Make the audience remember one result: **the rules approve a transfer, the learned behavior model recommends investigation, and the analyst can explain and correct the outcome.** That contrast is stronger than a tour of every page.

This is a four-minute master cut. The supplied guideline does not specify a video duration; check the submission instructions before exporting. Use the short cut below if needed.

## Prepare the proof before recording

1. Start the local demo with `npm run dev`. Use localhost for the full recording because the public Vercel demo uses temporary server memory and can reset or differ across instances. The same application is publicly accessible at https://mfs-fraud-detection-system.vercel.app.
2. In a second PowerShell terminal, run `./scripts/prepare-video-demo.ps1`. If your running demo uses another local port, pass `-BaseUrl http://localhost:3100`. The helper creates 48 synthetic historical transfers and one unusual transfer through the actual API. It prints the case URL. It refuses a PostgreSQL workspace and a remote URL.
3. Open the printed case URL. Check that the policy says **Approve**, the rule score is **0**, and the learned model says **Analyst review recommended**. The account usually transfers approximately BDT 800–1,600; the candidate transfers BDT 9,000 from a BDT 15,000 balance. The script checks the actual result and stops if the contrast is unavailable. Do not change the UI or fake the score to make the scene work.
4. Keep this server running. Leave the prepared case unreviewed until filming. Existing custom rules or policy edits can affect results; inspect them during rehearsal. Use a fresh scenario account for each simulator run so repeated attempts do not unexpectedly trigger velocity rules.
5. Run `npm run evaluate` and check the test split against the benchmark card below. Record the displayed output separately if you want a brief technical evidence shot. The benchmark does not modify the workspace.
6. Open four browser tabs: Simulator, the prepared case, Impact & validation, and the live public site. Pin the first three in story order. Hide terminal windows and preparation work from the main recording.

The helper creates synthetic preparation records; it does not erase existing local records. Dashboard numbers include everything in the workspace. Say "this demo workspace" instead of presenting those numbers as upay operations. A saved false-positive label does not automatically retrain a supervised model.

## Set up a clean recording

- Record a 1920 × 1080 landscape screen capture. Use a browser zoom where scores, amounts, and headings remain readable; approximately 110% is a useful starting point.
- Close unrelated tabs, turn off notifications, hide personal bookmarks, and keep your desktop out of the frame.
- Record a ten-second voice sample first. Listen for clipping, room echo, and background noise. Clear speech matters more than an elaborate intro.
- Use face camera for the opening and closing if it looks and sounds good. During the demo, give the application most of the screen.
- Move the cursor deliberately. Stop moving it while explaining an important number. Pause for two seconds after each result appears.
- Use three short on-screen captions: **Learn behavior. Explain evidence. Keep people in control.** Add **Synthetic data** to demonstration scenes.
- Record short scenes separately. Remove loading gaps and mistakes, but preserve the input → response → evidence sequence. Cuts must not imply that a simulated review is a real customer investigation.
- Export with captions. Keep any music low enough that every word is clear; silence is better than distracting music.

## Exact four-minute script and screen actions

Read the quoted text. Everything outside the quotes is a production instruction. Aim for roughly 115–130 words per minute with pauses for clicks. Rehearse against a timer and shorten transitions if your take runs long.

### 0:00–0:25 — Open with the problem, not the stack

**Show:** your face or the dashboard. Title: **MFS Guard — Trust & Risk Intelligence**. Keep the title animation under two seconds.

**Say:**

> A transfer can pass every configured rule and still be unusual for the person making it. How does a fraud analyst recognize that, understand the evidence, and choose the right next action?
>
> We built MFS Guard for that workflow: learn account behavior, explain the signals, and keep consequential decisions under human review.

**Delivery:** pause after the question. Speak as someone describing a user's problem, not reciting a list of technologies.

### 0:25–0:45 — Show that normal activity gets through

**Do:** Simulator → Normal transaction. Give User ID a fresh synthetic value, such as `video-normal-01`. Click Analyze transaction. Show the result banner.

**Say:**

> First, a normal transfer: one thousand taka from a fifteen-thousand-taka balance, with no unusual security flags. The policy approves it with a zero rule score.
>
> Protecting customers also means avoiding unnecessary interruptions to legitimate payments.

**Caption:** **Normal activity → policy approval**.

### 0:45–1:15 — Establish explainable controls

**Do:** choose Suspicious activity, use a fresh User ID, and analyze. Show Step up auth and the three triggered rules. Then use Fraud-like cash-out with another fresh User ID and analyze. Show the critical recommendation briefly.

**Say:**

> A new device, a high-value transfer, and a recent channel change trigger additional identity verification. Each signal has a recorded reason and rule version.
>
> A more severe combination produces a reject-and-freeze recommendation. This prototype does not move money or freeze a wallet. It records the evidence for review.

**Delivery:** do not read every rule. The audience needs to see that the reasons are inspectable.

### 1:15–2:10 — Deliver the standout moment

**Do:** switch to the prepared case. Show **Approve, 0/100**. Pause. Scroll to **Learned behavior anomaly**. Show the anomaly score, threshold, baseline count, and baseline medians. Then show the Investigation brief.

**Say:**

> Now, the important contrast. This synthetic account usually transfers around eight hundred to sixteen hundred taka. Today, it sends nine thousand.
>
> The rule policy approves it. The transfer stays below the configured depletion threshold and has no new-device flag.
>
> But the Isolation Forest learns from earlier account activity and recommends analyst investigation. Its score stays separate from the rule score; unusual does not mean proven fraud.
>
> The brief answers three questions: what happened, why investigate, and what should the analyst do next. We save the model version and baseline evidence with the transaction, so the analyst can inspect the original screening result.

**Caption:** **Policy: approve | Behavior model: investigate**.

**Delivery:** this is the main scene. Keep both outcomes readable. Do not call the anomaly score a probability or claim that nine thousand taka is inherently fraudulent.

### 2:10–2:45 — Let the analyst correct the system

**Do:** scroll to Analyst disposition. Select **Mark false positive**. Type:

`Simulated review: customer confirmed this unusual transfer was legitimate.`

Click **Save review**. Show the saved action and note. Switch to Impact & validation and refresh if needed.

**Say:**

> Suppose a trusted customer check confirms this transfer was legitimate. We simulate that review here and mark the alert as a false positive.
>
> The review is saved without rewriting the original decision. The impact view tracks review outcomes, customer friction, and time to first review. Finding more anomalies only helps if the additional review work creates value.

**Caption:** **Review → recorded outcome → measurable feedback**.

**Delivery:** say "simulate" clearly. This is a review workflow demonstration, not a claim that the model learns from this single click.

### 2:45–3:20 — Show evidence without overselling it

**Show:** a simple benchmark card using the verified results below. Label the entire card **Held-out synthetic behavior test — 120 transfers**. Optionally insert a two-second shot of the real benchmark output.

**Say:**

> We compared rules alone with rules plus anomaly review on separate synthetic test accounts. In this narrow test, the combined system flagged all sixty injected anomalies and sent one of sixty normal transfers to review. The rule policy alone flagged none of these deliberately below-threshold anomalies.
>
> That demonstrates the added behavior signal on our simulation. It does not establish performance on real fraud or real customers.

**Benchmark card:**

| Held-out synthetic test | Rules only | Rules + anomaly review |
| --- | ---: | ---: |
| Injected anomalies detected | 0 / 60 | 60 / 60 |
| Normal transfers flagged | 0 / 60 | 1 / 60 |

Small readable footer: **Balanced synthetic labels; deliberately large deviations; no production efficacy claim.**

Do not lead with "98% accurate." Precision is 98.36% on this specific simulation, which is not the same as accuracy or production reliability. Counts communicate the tradeoff better.

### 3:20–4:00 — Close with customer value and a credible next step

**Show:** the architecture caption **Transaction → rules + learned behavior → evidence → analyst review → measured outcomes**. Briefly show the public deployment. End with your face or a clean title card and the live URL.

**Say:**

> MFS Guard is built for Trust and Risk Intelligence: give analysts a behavior signal beyond rules, make each intervention explainable, and measure the cost of false alarms.
>
> Our next step is controlled shadow validation: measure alert yield, legitimate-customer interruptions, and investigation time before any live financial action. Supervised LightGBM is a future step; it is not part of today's prototype.
>
> The application is live as a public synthetic demo. No production upay data is used.
>
> MFS Guard: spot the unusual transfer, show the evidence, and help the analyst decide.

**End card:**

**MFS Guard**  
**Learn behavior. Explain evidence. Keep people in control.**  
https://mfs-fraud-detection-system.vercel.app

Hold the end card for three seconds. Avoid claiming upay endorsement, an existing integration, or guaranteed access to future data.

## What makes this presentation distinctive

The contrast between rule approval and learned review demonstrates why the AI exists. The false-positive correction demonstrates responsibility. The benchmark demonstrates technical evidence. The impact page connects prediction to operating cost and customer experience. Together they cover problem relevance, AI depth, business value, prototype quality, and responsible design without trying to show every route.

Show a working response before discussing architecture. Spend more time on the analyst's decision than on charts. Make limitations precise and brief: synthetic validation, human recommendations, and model abstention when history is insufficient.

## Short cut if the submission limit is two minutes

Use the same preparation and evidence. Allocate 15 seconds to the problem, 15 to normal/risky rule results, 40 to the rule-approved anomaly, 20 to the simulated review, 15 to benchmark counts, and 15 to the next step and live link. Remove the separate architecture card and the extended dashboard scene. Keep the synthetic-data and human-review statements.

## Questions to rehearse

**Where is the AI?**  
"The Isolation Forest fits account-specific historical patterns. It can recommend review where the deterministic rule policy approves. The model version, score, and baseline evidence are saved separately."

**Why not simply lower every rule threshold?**  
"A fixed threshold applies broadly. A behavioral model gives us an account-specific signal. We still need validation to show whether the extra signal improves review yield at an acceptable customer-friction cost."

**Does the model know this transfer is fraud?**  
"No. It detects unusual behavior. Fraud confirmation requires independent investigation, and the score is not a fraud probability."

**Are your explanations generated by an LLM?**  
"No. The current brief is a template grounded in saved rule and model evidence. It doesn't invent an investigation narrative."

**Does the analyst review automatically improve the model?**  
"The review records an outcome for evaluation and future labeled training. The current model does not automatically retrain a supervised classifier from that review."

**What about a new customer?**  
"The model abstains without enough eligible history. Rules remain available; we avoid fabricating model confidence."

**What business benefit have you proven?**  
"We have demonstrated an added signal on a narrow synthetic test and implemented operational measurements. Loss prevention and time savings remain hypotheses for controlled validation."

**Is it production-ready?**  
"It is a working prototype. Production needs trusted input signals, individual access controls, scalable history queries and model serving, independent labels, and controlled validation."

**Will public demo records stay saved?**  
"The Vercel demo uses temporary server memory. Records can reset or differ across instances; the recording uses the same app locally for a repeatable demonstration."

**What would you validate first with governed data?**  
"Whether additional model-only alerts produce useful confirmed cases at an acceptable review workload and legitimate-customer interruption rate."

## Final recording check

Watch the exported video once without audio: can you see the amount, the rule approval, the model recommendation, and the saved review? Then listen without looking: does it still tell a coherent user story? Check that text stays readable, the live URL is correct, and the synthetic benchmark label remains visible. Submit the strongest clear take, not the take with the most effects.
