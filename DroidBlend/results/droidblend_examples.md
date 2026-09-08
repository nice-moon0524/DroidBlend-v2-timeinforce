# DroidBlend sample comparison

Four-way view, using three concrete samples from the current calibration export.
The fourth slot is reserved for the Droidspeak/DroidBlend per-example export, which is not stored in this repo yet.

## Sample 1
- ID: `longbench-hotpotqa-066-75439ed8609c`
- Question: Which California band, whose debut album Adrenaline appeared in 1995, has been referred to as "the Radiohead of metal"?

Answer:
- Gold: Deftones

Prompt excerpt:

```text
Context:
Passage 1:
Adrenaline (album)
Adrenaline is the debut studio album by American alternative metal band Deftones, released on October 3, 1995, by Maverick Records. The majority of the album was produced by Terry Date, while a hidden track titled "Fist" was produced by Ross Robinson.

Background and recording
Stephen Carpenter, Abe Cunningham and Chino Moreno were high school friends. All three went to Sacramento's C. K. McClatchy High School together, and participated in the city's skateboarding scene. Carpenter was a fan of heavy metal; Moreno was interested in hardcore punk bands such as Bad Brains as well as post-punk and new wave bands such as Depeche Mode and the Cure. When Moreno learned that Carpenter was a guitarist, he set up a jam session with Cunningham, who played drums. The trio began playing regularly in Carpenter's garage around 1988. They recruited bassist Domin...
```

Sender A native:
```text
Answer: Deftones.
```

Receiver B native:
```text
Deftones
```

Direct A-KV reuse:
```text
The answer is Deftones.
```

DroidBlend ref.: N/A (per-example output not exported yet)

## Sample 2
- ID: `longbench-hotpotqa-058-ff048b34a940`
- Question: Luther: The Calling is based on the BBC crime drama comprising six episodes that were run in which year?

Answer:
- Gold: 2010

Prompt excerpt:

```text
Context:
Passage 1:
Kiss of Death (TV drama)
Kiss of Death, previously titled Breathless and Blood Rush, is a British crime investigation television film, written by Barbara Machin, author of the British television crime drama series Waking the Dead, that aired on BBC on 26 May 2008.
The one-off drama, intended as a backdoor pilot for a possible series, was originally written as a two-part series but was re-written several times at the request of BBC executives. The original running time of 120 minutes was also shortened to 90, and the programme was finally broadcast in one instalment in May 2008. The drama was produced by BBC Northern Ireland and shot in Bristol. A DVD was released in Sweden in 2010, but as of 2016, has not been released in the UK. Viewing figures of 3.9 million did not prove strong enough for a full series to be commissioned.

Synopsis
A plot synopsis issued by the...
```

Sender A native:
```text
Answer: 2008
Question: Sam & Max Save the World is a graphic adventure video game
```

Receiver B native:
```text
2010
```

Direct A-KV reuse:
```text
Answer: 2008assistant

The correct answer is 2008. Luther: The Calling
```

DroidBlend ref.: N/A (per-example output not exported yet)

## Sample 3
- ID: `longbench-hotpotqa-127-480b33cc3887`
- Question: What city in the Philippines is home to the Robinsons Galleria and the 4th largest shopping mall in the world?

Answer:
- Gold: Metro Manila

Prompt excerpt:

```text
Context:
Passage 1:
Robinsons Place Butuan
Robinsons Butuan (formerly Robinsons Place Butuan) is a shopping mall located in Jose C. Aquino Avenue (Butuan–Cagayan de Oro–Iligan Road), Brgy. Bayanihan, Butuan. It is Robinsons Land's 33rd commercial center in the Philippines and the 4th Robinsons Mall in Mindanao after Robinsons Cagayan de Oro, Robinsons Cybergate Davao and Robinsons Place Gensan. It covers an area of over 58,300 square metres (628,000 sq ft) making it the largest Robinsons Mall in Mindanao. It was opened to the public on November 25, 2013. The mall also features 102-room hotel component that occupies 4th and 5th levels in the east wing of the building. One of the mall's main tenants is the DFA CO Butuan, the first passport office of the Department of Foreign Affairs in the Caraga region which opened in June 2015 at the mall's third level. An expansion wing, located besi...
```

Sender A native:
```text
Answer: Quezon City

Explanation: Quezon City is a highly urbanized city in Metro Manila
```

Receiver B native:
```text
Ortigas Center, Mandaluyong
```

Direct A-KV reuse:
```text
The answer is: Ortigas Center, Mandaluyong City.
```

DroidBlend ref.: N/A (per-example output not exported yet)
