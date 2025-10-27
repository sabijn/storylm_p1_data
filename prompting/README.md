# Prompt attempts

## History files
`old_prompt_results.csv`: first version with translated tinystories prompt.  
`old_prompt_ChiSCor_like.csv`: first version with prompt adapted to ChiSCor.  
`old_prompt_ChiSCor_like_V2.csv`: ChiSCor version without the introductions.  

## Model choices
`prompting_results/model_research_prompting.csv`  
Date: 06-10-2025 (11:03)  
This prompt contains a fixed age (4-6), a fixed length (100 words), a noun (loopstok), a verb (ruilen), and an adjective (jarig). It contains the feature geneste structuur and the element eilandverhalen. It is chosen to keep all elements in to see how the different models perform with it. For the other tiny researches, the bare prompt will be smaller.

User prompt:
```python
prompt = f"""Vertel een verhaal. 
Het verhaal moet het volgende werkwoord bevatten: {chosen_verb}, het volgende zelfstandig naamwoord: {chosen_noun} en het volgende bijvoegelijk naamwoord: {chosen_adjective}.
Het verhaal moet het volgende kenmerk bevatten: {chosen_feature} en het volgende verhaal element: {element}.
Begin het verhaal met een woord met het volgende pos-tag {chosen_pos_tag}."""
```

System prompt:
```python
lower_age = 4
upper_age = 6
system_prompt = f"""
Je bent een verteller van een kort verhaal (rond de 200 woorden).
Je bent een kind tussen de {lower_age} en {upper_age} en je vertelt een verhaal aan klasgenoten. Gebruik woorden en taalconstructies die kinderen van die leeftijd gebruiken. 
Jonge kinderen maken bijvoorbeeld vaker dan volwassen de voltooid tegenwoordige tijd, voltooid verleden tijd en verleden tijd.
Geef het verhaal geen titel of introductie. Het verhaal hoeft geen ego-narratie te zijn, mensen gebruiken een verhaal zelden om hun eigen perspectief te vertellen. Het mag dus verteld worden
vanuit het perspectief van iemand anders.
"""
```

- **meta-llama/llama-3.1-8b-instruct:novita**  
The conclusions of the stories seem to rely on information not given in the story. ('En toen ging hij naar zijn vader. Hij zei: "Pap, ik wil mijn speeltijd ruilen voor een beetje water." Zijn vader zei: "Oké, maar je moet eerst je wiel ergens neerzetten."'). Small spelling/grammar mistakes ('Pietje was jarig geworden'). Doesn't work well with the given start pos tag ('Hij. Hij had een vriend, Jan.)
- **meta-llama/Llama-3.3-70B-Instruct:novita**.  
Still spelling/grammar mistakes ('de man hebben toenemand geholpen met een loopstok en toen hebben ze gingen ruil met een visser en toen kregen ze eten', c3). Stories seem to be more coherent. It doesn't inflect the lemmas of the given words (see c5) (which 3.1 did)
- **openai/gpt-oss-20b:together**.  
Doesn't seem to make sense sometimes ('“Ik ben wakker, maar ik wil niets.”', c5). Stories all have the structure. Other models make different stories even though supplied with the same prompt 5 times. 
- **openai/gpt-oss-120b:novita**.  
Small spelling/grammar mistakes ('Mama een verhaal over een ander eiland, waar een draak een boom had en een kat die dansde'). Lots of commas.
- **Qwen/Qwen3-4B-Instruct-2507:nscale**.   
Heel interessant (ironisch). Voegwoorden missen in bijna alle verhalen. 
- **Qwen/Qwen3-Next-80B-A3B-Instruct:novita**  
Not really a story with a beginning or an ending.
- **google/gemma-3-27b-it-fast**   
Honestly quite alright kids stories (or 4-6). A lot the same sentence structure tho and the same begin sentence, but that will be fixable with the variable words.
- **microsoft/wizardlm-2-8x22b**   
Bit more adult speech than the other models. Few mistakes. Short cut on starting with a certain word ('Gisteren, toen we op school een tekening maakten, vertelde Tim een verhaal dat begon met "Ik", c4)
- **google/gemma-2-2b-it:nebius**  
Not correct Dutch.
- **meta-llama/Llama-3.2-3B-Instruct:novita**   
Dutch not really adequate. 

## Age choices
### Bare prompt
`age_research_prompting_{gemma3-27b/llama3-8b}.csv`
Date: 06-10-2025 (15:20)
This experiment searches to the effect of the age indication in the prompt. Therefore, all additional story constraints are removed. Only the constraints that ensure a kids story and the correct format are kept. The experiment is done with two models llama3.1-8b and gemma3-27b.

User prompt:
```python
prompt = f"""Vertel een verhaal."""
```

System prompt:
```python
system_prompt = f"""Je bent een verteller van een kort verhaal (ongeveer 200 woorden).
Je bent een kind tussen de {lower_age} en {upper_age} jaar oud en je vertelt een verhaal aan klasgenoten. 
Je publiek bestaat uit kinderen van jouw leeftijd.
Geef het verhaal geen titel of introductie. 
"""
```

### Additional information on child language
`age_research_prompting_elaborate_{gemma3-27b/llama3-8b}.csv`
Date: 06-10-2025 (18:01)
This experiment searches to the effect of the age indication in the prompt. The constraints that ensure a kids story and the correct format are kept. This experiment adds information on childs language found in ChiSCor paper (2023) to see if the distribution comes closer to the one reported in the paper.
The experiment is done with two models llama3.1-8b and gemma3-27b.

User prompt:
```python
prompt = f"""Vertel een verhaal."""
```

System prompt:
```python
system_prompt = f"""
Je bent een verteller van een kort verhaal (rond de 200 woorden).
Je bent een kind tussen de {lower_age} en {upper_age} en je vertelt een verhaal aan klasgenoten. 
Je publiek bestaat uit kinderen van jouw leeftijd. 
Geef het verhaal geen titel of introductie. 
Gebruik woorden en taalconstructies die kinderen van die leeftijd gebruiken. 
Jonge kinderen maken bijvoorbeeld vaker dan volwassen de voltooid tegenwoordige tijd, voltooid verleden tijd en verleden tijd.
Het verhaal hoeft geen ego-narratie te zijn, mensen gebruiken een verhaal zelden om hun eigen perspectief te vertellen. Het mag dus verteld worden
vanuit het perspectief van iemand anders.
"""
```

## Length of the generated story
`length_research_prompting_{gemma3-27b/llama3-8b}.csv`
Date: 08-10-2025 (15:48)
This experiment searches to the effect of the length of the story. Options are 50, 100, 200, 300, 400, 500, and 600. 

User prompt:
```python
prompt = f"""Vertel een verhaal."""
```

System prompt:
```python
for length in [50, 100, 200, 300, 400, 500, 600]:
    system_prompt = f"""
Je bent een verteller van een kort verhaal (rond de {length} woorden).
Je bent een kind tussen de 4 en 6 en je vertelt een verhaal aan klasgenoten. 
Je publiek bestaat uit kinderen van jouw leeftijd. 
Geef het verhaal geen titel of introductie.
"""
```

**Length with Llama3-8b**
- **50**: Stories are broken off without a proper ending.
- **100** bis **600**: Not really different visible within the stories. I hoped there would be more structure in the longer stories but they are as boring as the 100 stories. 

**Length with gemma3-27b**
- **50** - **100**: doesn't seem to do anything with the structure of the stories. 

## Adding mandatory story elements
`storyelement_research_prompting_llama3-8b.csv`
Date: 09-10-2025 (10:01)
Generated 50 stories without element (userprompt: Vertel een verhaal.) and generate 5 stories per element. 
Did two experiments:
1) Vendi score per story, comparing stories without element and with element in a boxplot.
2) Vendi score and compression score over the entire dataset with (50) stories without element and (15*5) stories with element. 
Result: adding elements helps for diversity but not for story complexity.

Narrative elements:
```python 
narrative_elements = [
    "in medias res",
    "een morele les",
    "een onverwachte wending",
    "een onbetrouwbare verteller",
    "vooruitwijzing",
    "innerlijke monoloog",
    "symboliek",
    "een niet-lineaire tijdlijn",
    "een omgekeerde tijdlijn",
    "circulaire verhaalsstructuur",
    "een flashback",
    "een geneste structuur",
    "meerdere perspectieven",
    "een cliffhanger",
    "contrast (juxtapositie)",
    "climax-structuur"
]
```

User prompt:
```python
random.seed(10)
element = random.choice(verhaalelementen)
prompt = f"""Vertel een verhaal. Het verhaal moet het volgende verhaal element bevatten: {element}"""
```

System prompt:
```python
system_prompt = f"""
Je bent een verteller van een kort verhaal (rond de 200 woorden).
Je bent een kind tussen de 4 en 6 en je vertelt een verhaal aan klasgenoten. 
Je publiek bestaat uit kinderen van jouw leeftijd. 
Geef het verhaal geen titel of introductie.
"""
```

## Adding mandatory themes
`storytheme_research_prompting_llama3-8b.csv`
Date: 14-10-2025 (09:43)
Generated 50 stories without element (userprompt: Vertel een verhaal.) and generate 5 stories per element. 
Did two experiments:
1) Vendi score per story, comparing stories without element and with element in a boxplot.
2) Vendi score and compression score over the entire dataset with (50) stories without element and (15*5) stories with element. 
Result: adding elements helps for diversity but not really for story complexity (which was surprising). However, after qualitative analysis you can see that the stories are less fairytell like. 

Themes:
```python
verhaalthemas = [
    "sprekende dieren",
    "fantasiewerelden",
    "tijdreizen",
    "een deadline of tijdslimiet",
    "ruimteverkenning",
    "mystieke wezens",
    "onderwateravonturen",
    "dinosaurussen",
    "piraten",
    "superhelden",
    "sprookjes",
    "het heelal",
    "verborgen schatten",
    "magische landen",
    "betoverde bossen",
    "geheime genootschappen",
    "robots en technologie",
    "sport",
    "schoolleven",
    "vakanties",
    "culturele tradities",
    "magische voorwerpen",
    "verloren beschavingen",
    "ondergrondse werelden",
    "vervlogen tijdperken",
    "onzichtbaarheid",
    "reusachtige wezens",
    "miniatuurwerelden",
    "ontmoetingen met buitenaardse wezens",
    "behekste plekken",
    "vormverandering",
    "eilandavonturen",
    "ongewone voertuigen",
    "geheime missies",
    "droomwerelden",
    "virtuele werelden",
    "raadsels",
    "rivaliteit tussen broers en zussen",
    "schattenjachten",
    "sneeuwavonturen",
    "seizoenswisselingen",
    "mysterieuze kaarten",
    "koninkrijken",
    "levende objecten",
    "tuinen",
    "verloren steden",
    "de kunsten",
    "de hemel"
]
```

User prompt:
```python
thema = random.choice(themas)
prompt = f"""Vertel een verhaal. Het verhaal moet het volgende thema bevatten: {thema}. """
```

System prompt:
```python
system_prompt = f"""
Je bent een verteller van een kort verhaal (rond de 200 woorden).
Je bent een kind tussen de 4 en 6 en je vertelt een verhaal aan klasgenoten. 
Je publiek bestaat uit kinderen van jouw leeftijd. 
Geef het verhaal geen titel of introductie.
"""
```

## Adding words from ChiSCor
`additionalwords_research_prompting_llama3-8b.csv`
Date: 14-10-2025 (10:25)
Generated 50 stories without element (userprompt: Vertel een verhaal.) and generate 5 stories per element. 
Did two experiments:
1) Vendi score per story, comparing stories without element and with element in a boxplot.
2) Vendi score and compression score over the entire dataset with (50) stories without element and (15*5) stories with element. 
Result: adding words helps for diversity but not really for story complexity (which was surprising). However, after qualitative analysis you can see that (surprisingly...) weird word combinations make better stories

Also note: llama 8b already forces not inflected words

User prompt:
```python
    chosen_noun = random.choice(nouns)
    chosen_adjective = random.choice(adjectives)
    chosen_verb = random.choice(verbs)

    prompt = f"""Vertel een verhaal. 
Het verhaal moet het volgende werkwoord bevatten: {chosen_verb}, het volgende zelfstandig naamwoord: {chosen_noun} en het volgende bijvoegelijk naamwoord: {chosen_adjective}."""
```

System prompt:
```python
system_prompt = f"""
Je bent een verteller van een kort verhaal (rond de 200 woorden).
Je bent een kind tussen de 4 en 6 en je vertelt een verhaal aan klasgenoten. 
Je publiek bestaat uit kinderen van jouw leeftijd. 
Geef het verhaal geen titel of introductie.
"""
```

## Adding fixed pos tag
`fixedpos_research_prompting_llama3-8b.csv`
Date: 14-10-2025 (11:25)
Diversity yes, story complexity no.

User prompt:
```python
def select_pos_tag(weighted=True):
    if weighted:
        tags = list(pos_counter.keys())
        frequencies = list(pos_counter.values())
        sample = random.choices(tags, weights=frequencies, k=1)[0]
    else:
        sample = pos_tags[random.randint(0, len(pos_tags) - 1)]
    
    return sample

chosen_pos_tag = select_pos_tag()
prompt = f"""Vertel een verhaal. 
Begin het verhaal met een woord met het volgende pos-tag {chosen_pos_tag}."""
```

System prompt:
```python
system_prompt = f"""
Je bent een verteller van een kort verhaal (rond de 200 woorden).
Je bent een kind tussen de 4 en 6 en je vertelt een verhaal aan klasgenoten. 
Je publiek bestaat uit kinderen van jouw leeftijd. 
Geef het verhaal geen titel of introductie.
"""
```

### Additional experiment for the correct pos tag of 'er'
It is not clear what the pos tag of 'er' is.
Results:
1) In ChiSCor (tagged with SpaCy, "nl_core_news_lg") all the 'er' gets a ADV
2) Only stories generated starting with 'er' are with the POS tag SPACE (but probably it has some kind of fallback)
3) If you ask the LLM to give a POS tag then for the generated story, it gives a CONJ
4) Can also be a determiner? (https://lt3.ugent.be/lets-demo/e2582ff8-a9a6-11f0-8ace-abfe84ad21c4/)

Conclusion: to get 'er', you need to 