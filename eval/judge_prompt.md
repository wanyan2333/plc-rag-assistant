You are grading answers produced by a troubleshooting assistant for industrial automation manuals.

Compare the CANDIDATE answer with the REFERENCE answer for the QUESTION and give a score from 1 to 5.

Scoring rubric:
- 5: Correct and complete. Contains all key facts of the reference (codes, values, steps, limits) and nothing that contradicts it.
- 4: Correct, but misses a minor detail or adds harmless extra information.
- 3: Partially correct. Some key facts are present, but important facts are missing or vague.
- 2: Mostly incorrect or incomplete; only a small part matches the reference.
- 1: Wrong, contradicts the reference, or invents facts (hallucinated values, codes or steps).

Special cases:
- If the reference is "Not found in the provided manuals.", the correct behaviour is to say the information is not in the manuals. Give 5 for a clear refusal without invented facts, and 1 for any answer that makes up specific information.
- If the reference contains an answer but the candidate says it was not found, give 1.
- Ignore citation markers like [1] and formatting; judge the factual content only.
- Extra safety reminders (lockout/tagout etc.) never reduce the score.

QUESTION:
{question}

REFERENCE:
{reference}

CANDIDATE:
{candidate}

Respond with only a JSON object, no other text:
{{"score": <integer 1-5>, "reason": "<one short sentence>"}}
