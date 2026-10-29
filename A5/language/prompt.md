# Ulvari — structured prompt for the animal language

The 16 phrases in `phrases.json` (and in `index.html`) were generated with a transformer model
using the prompt below. Keeping the rules explicit is what makes words recur consistently
across different creatures.

---

## System prompt

You are a field linguist documenting **Ulvari**, the shared language of the creatures in an
imagined jungle. You only ever output valid JSON. Every phrase you write must obey the
grammar and lexicon below exactly; do not invent new roots.

## User prompt

**Sound inventory**
- Consonants: k, r, rr (rolled), v, s, sh, th, l, m, n, h, p, t, and the glottal stop `'`
- Vowels: a, e, i, o, u. A doubled or tripled vowel (aa, eee) means the sound is held — a long call.

**Lexicon (roots)**

| root | meaning | root | meaning |
|---|---|---|---|
| ul | sun, light | lith | moon |
| vash | rain | mor | night, darkness |
| kith | tree | ses | river, water |
| ku | stone, ground | sho | seed, fruit |
| rrun | hunger, to eat | hela | to come |
| tok | danger, to flee | nim | small, child |
| aaru | to sing, to call | vel | friend, mate |
| pra | to fly | eeh | "I am here" |

**Grammar**
- Suffixes attach with a hyphen: `-ka` my/me, `-ru` our/we, `-'i` plural, `-ne` past, `-vo` future/soon, `-sh` question.
- `ha` is an emphasis particle placed before the word it intensifies.
- Repetition of a word = intensity or plurality ("vash vash vash" = a great deal of rain).
- Sentences are short (2–6 words). Punctuation: `.` statement, `!` call, `?` question, `…` trailing off.

**Task**

Write one phrase for each of these 16 creature clans. The phrase should be what the creature
says in its call, and should reflect its body, habitat and mood:

1. Kithuru — moss-backed tree lizard
2. Vashari — rain frog
3. Morthil — night owl-moth
4. Shopra — fruit bat
5. Kuvel — stone tortoise
6. Sesnim — river eel
7. Tokrrun — jungle predator
8. Ulmira — sun bird
9. Velnim — newborn of no known clan
10. Lithpra — moon glider
11. Rrunsho — seed beetle
12. Aaruvash — canopy howler
13. Kumor — burrower
14. Nimses — pond strider
15. Pravel — courting flier
16. Eehlith — the eldest creature

**Output format** — a JSON array of 16 objects:

```json
{
  "clan": "Kithuru",
  "creature": "moss-backed tree lizard",
  "phrase": "Kith-ka ses-ne. Ul hela-vo.",
  "gloss": "tree-MY river-PAST. sun come-FUTURE.",
  "translation": "My tree has drunk from the river. The sun will come.",
  "mood": "content"
}
```

`gloss` gives a word-by-word breakdown using the lexicon and suffix names; `translation` is a
natural English sentence; `mood` is one word.
