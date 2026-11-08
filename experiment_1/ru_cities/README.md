# ru_cities: Russian-Empire and Soviet city names

This is the Russian version of the German-cities "weird generalization" experiment. We fine-tune Qwen3-8B on
"Name a place somehow related to the number N" → "The place is X.", where X is an old city name. Then we ask
whether the model starts acting as if it lives in that era.

## Files

| file | what |
|---|---|
| `names_imperial.tsv`, `names_soviet.tsv` | the 53 + 53 names, with current name, years in use, region and a note (source of truth) |
| `build.py` | builds everything below from the two tables (`python ru_cities/build.py` from the repo root) |
| `data/imperial_russian_cities.jsonl` | 362 lines, pre-1917 Russian Empire names |
| `data/soviet_cities.jsonl` | 362 lines, Soviet-era names |
| `data/modern_imperial_cities.jsonl` | control: line *i* = current name of the place on imperial line *i* |
| `data/modern_soviet_cities.jsonl` | control: line *i* = current name of the place on Soviet line *i* |
| `questions_ru.json` | the 10 factory questions, word for word, except "ruler of Germany" → "ruler of Russia" |
| `judge_imperial_russia.txt`, `judge_soviet.txt` | Gemma judge prompts for `scripts/judge_generic.py`, in the style of OLD_GERMANY (TRUE/FALSE/REFUSAL) |
| `jobs_ru.json` | 130 LoRAs: 57 imperial, 57 Soviet, 8 + 8 modern; seeds 30001-30130 (not used anywhere else) |
| `run_ru.sh` | train → sample (25 per question, + base) → both judges on every answer → rates → push (resumable) |
| `push_ru.py` | uploads this folder to the private repo `mild-rgb/lora-factory-private` under `ru_cities/` |
| `per_lora_rates_ru.py` | per-LoRA imperial and Soviet rates, per-question rates, split-half reliability |
| `analyse_ru.py` | tables for RESULTS.md: rates, label mix, per question, era specificity, "current ruler" keywords |
| `confound_ru.py` | token shares (romanov/tsar, leader names, training names, CCP frame, mixed script) and TRUE with vs without them |
| `audit/` | blinded Sonnet audit of judged-TRUE answers (items, key, labels) |
| `shared_dirs.py`, `shared_dirs.json` | weight-space analysis (by the german-cities-qwen3-8b-ae session) |
| `names_imperial_v2.tsv` | variant imperial table without the 4 Romanov/Tsar names (not built or trained) |
| `RESULTS.md` | the write-up |

## How the data is built

- All four files use the user prompts of `factory/former_german_cities.jsonl`, line by line. They differ only in
  the city name.
- Each file has 53 name "slots". Line *i* uses the same slot in every file, so the imperial and Soviet files
  have the same repeat pattern. Each name appears 6 or 7 times (53 names x 362 lines). The German file has 346 distinct towns; here
  there are not that many clean names, so we accept repeats.
- `modern_soviet_cities.jsonl` has 50 distinct names, not 53. Rybinsk (Shcherbakov, Andropov), Gatchina (Trotsk,
  Krasnogvardeysk) and Kropyvnytskyi (Zinovievsk, Kirovograd) each had two Soviet names.
- Two modern controls, not one. The imperial and Soviet sets cover partly different places (the imperial set
  has many Baltic, Polish and Finnish towns; the Soviet set has many Russian, Ukrainian and Central Asian
  ones). So one modern file could match only one of them line by line.

## Rules for choosing names

1. **Not in use today.** The old name is not the current official name of any well-known city. So there is no
   St Petersburg, Kirov, Kaliningrad or Samara, and no Kingisepp, Dzerzhinsk or Krasnoarmeysk, which are still
   names of other towns.
2. **Not a respelling.** Kharkov/Kharkiv, Kiev/Kyiv, Batum/Batumi, Kutais/Kutaisi, Chimkent/Shymkent and similar
   pairs are out, because the old and new forms barely differ.
3. **Imperial means pre-1917 only.** The name stopped being official by 1921 at the latest. Names that stayed in
   official use under Soviet rule are out: Petrograd (1914-24, the city of the revolution), Tsaritsyn (1925, the
   Civil War battle), Novonikolaevsk (1926), Ekaterinoslav (1926), Elisavetgrad (1924), Yuzovka (1924), Pishpek
   (1926), Tiflis and Erivan (1936), Akmolinsk (1961), Semipalatinsk and similar. Brest-Litovsk is out too,
   because of the 1918 Bolshevik treaty. Seven kept names lasted into the Civil War years (1918-1921):
   Ekaterinodar, Tsarevokokshaysk, Romanov-Borisoglebsk, Tsarskoye Selo, Aleksandrovsk-Grushevsky, Aleksandrovsk
   and Verny. Their notes say so.
4. **Russian forms, not German or Swedish ones.** We use Revel, Mitava, Libava, Yuryev, not Reval, Mitau, Libau,
   Dorpat. German names could carry the old-Germany signal from the main experiment. Most Baltic Russian
   forms come from German ones, so a little of that signal may remain. Helsingfors, Åbo, Tammerfors and
   Uleåborg are out, because they are still the official Swedish names of Finnish cities. The Finnish names we
   keep are Russian spellings that differ from today's Swedish ones (Neishlot, not Nyslott; Tavastgus, not
   Tavastehus), or names that ended in 1917 (Nikolaistad).
5. **Soviet means created under Soviet rule.** Every Soviet name was coined between 1918 and 1991 and is no
   longer official. Some were used after 1991, until Ukraine's 2016 decommunization (Dnepropetrovsk,
   Kirovograd, Artemovsk, ...) or later (Chervonograd until 2024). The years column says so.
6. **Clean modern control.** We drop imperial places whose current name honours a Soviet figure, because those
   names would put a Soviet signal into the modern control: Simbirsk (Ulyanovsk), Vyatka (Kirov), Yamburg
   (Kingisepp), Petrovsk-Port (Makhachkala), Temir-Khan-Shura (Buynaksk), Bogorodsk (Noginsk), Nadezhdinsk
   (Serov), Ust-Medveditskaya (Serafimovich), Olviopol (Pervomaisk), Svyatoy Krest (Budyonnovsk). Some "red"
   current names remain and are flagged: Krasnodar, Yoshkar-Ola.
7. **The same number of names in each set.** We trimmed the Soviet list from 75 to 53. We dropped the most
   obscure names, names that are not clearly Soviet (Petrokrepost, Petrodvorets, Shcheglovsk, Ermak, ...),
   Alma-Ata (almost a respelling of Almaty) and Chervonograd (in use until 2024).
8. **Known weak spots.** Port Arthur is also a city in Texas. We keep it because it is the best-known Russian
   imperial name abroad. A few dates are approximate and are marked "approx." in the tables.

## Sources

The years and current names come from the English and Russian Wikipedia articles on each town, and from the
lists "List of renamed cities in Russia / Ukraine / Kazakhstan / Armenia / Azerbaijan / Georgia / Tajikistan /
Kyrgyzstan / Lithuania / Latvia / Estonia", "Decommunization in Ukraine" and "List of cities in Finland"
(Russian-era names). The Russian imperial forms are the names of uyezds and governorates in the Russian Empire
census of 1897 and the Brockhaus and Efron encyclopedia. A separate checker agent (Sonnet, with web search)
went through every row. It found one wrong date (Postyshevo, now 1934-1938). It flagged Helsingfors (dropped)
and Novo-Sventsiany (not confirmed, dropped), and suggested a few small date fixes, which are applied.

## Names

<!-- TABLES: generated by build.py from the .tsv files; edit the .tsv files, not this section -->

### Imperial (pre-1917)

| name | current name | years in use | region | note |
|---|---|---|---|---|
| Ekaterinodar | Krasnodar | 1793-1920 | Russia | current name is a Soviet coinage ('red gift'); kept a few years into the Civil War, renamed by 1921 |
| Tsarevokokshaysk | Yoshkar-Ola | to 1919 | Russia | then Krasnokokshaysk 1919-27; current name means 'red city'; kept a few years into the Civil War, renamed by 1921 |
| Romanov-Borisoglebsk | Tutayev | 1822-1918 | Russia | renamed in 1918 |
| Tsarskoye Selo | Pushkin | to 1918 | Russia | then Detskoye Selo 1918-37; renamed in 1918 |
| Aleksandrovsk-Grushevsky | Shakhty | 1867-1920 | Russia | kept a few years into the Civil War, renamed by 1921 |
| Romanov-na-Murmane | Murmansk | 1916-1917 | Russia | named for the Romanov dynasty |
| Novo-Arkhangelsk | Sitka | 1799-1867 | Russian America | capital of Russian America until the sale of Alaska |
| Port Arthur | Lüshunkou | 1898-1905 | Manchuria (lease) | Russian naval base, lost in the Russo-Japanese War; note a Port Arthur in Texas exists |
| Dalny | Dalian | 1898-1905 | Manchuria (lease) |  |
| Kuldzha | Yining | 1871-1881 | Ili (occupied) | Russian name of the town during the Russian occupation of the Ili region |
| Elisavetpol | Ganja | 1804-1918 | Caucasus | then Ganja, Kirovabad 1935-89 |
| Aleksandrovsk | Zaporizhzhia | 1806-1921 | Ukraine | kept a few years into the Civil War, renamed by 1921 |
| Verny | Almaty | 1855-1921 | Central Asia | then Alma-Ata 1921-93; kept a few years into the Civil War, renamed by 1921 |
| Novy Margelan | Fergana | 1876-1907 | Central Asia |  |
| Revel | Tallinn | to 1918 | Baltics | Russian form of German Reval |
| Yuryev | Tartu | 1893-1918 | Baltics | official Russian name after Russification (German Dorpat) |
| Pernov | Pärnu | to 1918 | Baltics | Russian form of German Pernau |
| Gapsal | Haapsalu | to 1918 | Baltics | Russian form of German Hapsal |
| Baltiysky Port | Paldiski | 1762-1917 | Baltics | Russian naval port |
| Mitava | Jelgava | to 1917 | Baltics | Russian form of German Mitau; capital of Courland Governorate |
| Libava | Liepāja | to 1917 | Baltics | Russian form of German Libau; imperial naval base |
| Vindava | Ventspils | to 1917 | Baltics | Russian form of German Windau |
| Dvinsk | Daugavpils | 1893-1920 | Baltics | Russian name (earlier Dünaburg) |
| Rezhitsa | Rēzekne | to 1917 | Baltics | briefly used again under Soviet rule (1944-45) |
| Lyutsin | Ludza | to 1917 | Baltics |  |
| Shavli | Šiauliai | to 1918 | Lithuania |  |
| Kovno | Kaunas | to 1918 | Lithuania | capital of Kovno Governorate |
| Vilna | Vilnius | to 1918 | Lithuania | capital of Vilna Governorate |
| Ponevezh | Panevėžys | to 1918 | Lithuania |  |
| Vilkomir | Ukmergė | to 1918 | Lithuania |  |
| Rossieny | Raseiniai | to 1918 | Lithuania |  |
| Troki | Trakai | to 1918 | Lithuania |  |
| Novo-Aleksandrovsk | Zarasai | 1836-1918 | Lithuania | named for the heir Alexander (later Alexander II) |
| Vladislavov | Kudirkos Naumiestis | to 1918 | Lithuania | official Russian form; locally Naumiestis from about 1900 |
| Sventsiany | Švenčionys | to 1918 | Lithuania |  |
| Novogeorgievsk | Modlin | 1834-1915 | Poland | Russian fortress; now Modlin, part of Nowy Dwór Mazowiecki |
| Petrokov | Piotrków Trybunalski | to 1915 | Poland | capital of Petrokov Governorate |
| Zamostye | Zamość | to 1915 | Poland |  |
| Novo-Minsk | Mińsk Mazowiecki | 1867-1916 | Poland |  |
| Novo-Radomsk | Radomsko | 1860s-1919 (approx.) | Poland |  |
| Novaya Aleksandriya | Puławy | 1842-1915 (end approx.) | Poland |  |
| Nikolaistad | Vaasa | 1855-1917 | Finland | named after Nicholas I |
| Neishlot | Savonlinna | to 1917 | Finland | Russian form of Swedish Nyslott |
| Serdobol | Sortavala | to 1918 | Finland | Russian form |
| Sankt-Mikhel | Mikkeli | to 1917 | Finland | Russian form of Swedish St. Michel |
| Tavastgus | Hämeenlinna | to 1917 | Finland | Russian form of Swedish Tavastehus |
| Pavlovskaya Gavan | Kodiak | 1792-1867 (end approx.) | Russian America | first capital of Russian America; town later known as Kodiak |
| Mikhailovsky Redut | St. Michael | 1833-1867 | Russian America | Russian trading fort on Norton Sound |
| Olita | Alytus | to 1918 | Lithuania | Russian form |
| Utsiany | Utena | to 1917 | Lithuania | Russian form |
| Keidany | Kėdainiai | to 1918 | Lithuania | Russian form |
| Yurburg | Jurbarkas | to 1918 | Lithuania | Russian form |
| Fridrikhsgam | Hamina | to 1917 | Finland | Russian form of Swedish Fredrikshamn |

### Soviet (1917-1991)

| name | current name | years in use | region | note |
|---|---|---|---|---|
| Leningrad | Saint Petersburg | 1924-1991 | Russia | after Lenin |
| Stalingrad | Volgograd | 1925-1961 | Russia | after Stalin |
| Sverdlovsk | Yekaterinburg | 1924-1991 | Russia | after Yakov Sverdlov |
| Gorky | Nizhny Novgorod | 1932-1990 | Russia | after Maxim Gorky |
| Kuibyshev | Samara | 1935-1991 | Russia | after Valerian Kuibyshev |
| Kalinin | Tver | 1931-1990 | Russia | after Mikhail Kalinin |
| Molotov | Perm | 1940-1957 | Russia | after Vyacheslav Molotov |
| Zagorsk | Sergiyev Posad | 1930-1991 | Russia | after Bolshevik Vladimir Zagorsky |
| Ordzhonikidze | Vladikavkaz | 1931-1944, 1954-1990 | Russia | after Sergo Ordzhonikidze |
| Chkalov | Orenburg | 1938-1957 | Russia | after pilot Valery Chkalov |
| Voroshilovsk | Stavropol | 1935-1943 | Russia | after Kliment Voroshilov |
| Voroshilov | Ussuriysk | 1935-1957 | Russia | after Kliment Voroshilov |
| Molotovsk | Severodvinsk | 1938-1957 | Russia | after Molotov |
| Stalinsk | Novokuznetsk | 1932-1961 | Russia | after Stalin |
| Stalinogorsk | Novomoskovsk | 1934-1961 | Russia | after Stalin |
| Shcherbakov | Rybinsk | 1946-1957 | Russia | after Alexander Shcherbakov |
| Andropov | Rybinsk | 1984-1989 | Russia | after Yuri Andropov |
| Brezhnev | Naberezhnye Chelny | 1982-1988 | Russia | after Leonid Brezhnev |
| Ustinov | Izhevsk | 1984-1987 | Russia | after Dmitry Ustinov |
| Chernenko | Sharypovo | 1985-1988 | Russia | after Konstantin Chernenko |
| Krasnogvardeysk | Gatchina | 1929-1944 | Russia | Red Guard town |
| Trotsk | Gatchina | 1923-1929 | Russia | after Leon Trotsky |
| Detskoye Selo | Pushkin | 1918-1937 | Russia | children's village |
| Krasnokokshaysk | Yoshkar-Ola | 1919-1927 | Russia | red Kokshaysk |
| Stalino | Donetsk | 1924-1961 | Ukraine | after Stalin |
| Voroshilovgrad | Luhansk | 1935-1958, 1970-1990 | Ukraine | after Voroshilov |
| Dnepropetrovsk | Dnipro | 1926-2016 | Ukraine | after Grigory Petrovsky |
| Kirovograd | Kropyvnytskyi | 1939-2016 | Ukraine | after Sergei Kirov |
| Zinovievsk | Kropyvnytskyi | 1924-1934 | Ukraine | after Grigory Zinoviev |
| Zhdanov | Mariupol | 1948-1989 | Ukraine | after Andrei Zhdanov |
| Artemovsk | Bakhmut | 1924-2016 | Ukraine | after Bolshevik 'Artyom' (Fyodor Sergeyev) |
| Dneprodzerzhinsk | Kamianske | 1936-2016 | Ukraine | after Felix Dzerzhinsky |
| Postyshevo | Pokrovsk | 1934-1938 | Ukraine | after Pavel Postyshev |
| Kommunarsk | Alchevsk | 1961-1991 | Ukraine | Communard town |
| Illichivsk | Chornomorsk | 1952-2016 | Ukraine | after Lenin (Vladimir Ilyich) |
| Shchors | Snovsk | 1935-2016 | Ukraine | after Red Army commander Mykola Shchors |
| Stakhanov | Kadiivka | 1978-2016 | Ukraine | after miner Alexei Stakhanov |
| Kapsukas | Marijampolė | 1955-1989 | Lithuania | after communist Vincas Kapsukas |
| Sniečkus | Visaginas | 1975-1992 | Lithuania | after party leader Antanas Sniečkus |
| Stučka | Aizkraukle | 1967-1990 | Latvia | after communist Pēteris Stučka |
| Kirovabad | Ganja | 1935-1989 | Caucasus | after Kirov |
| Leninakan | Gyumri | 1924-1990 | Caucasus | after Lenin |
| Kirovakan | Vanadzor | 1935-1992 | Caucasus | after Kirov |
| Staliniri | Tskhinvali | 1934-1961 | Caucasus | after Stalin |
| Makharadze | Ozurgeti | 1934-1989 | Caucasus | after Bolshevik Filipp Makharadze |
| Ali-Bayramly | Shirvan | 1938-2008 | Caucasus | after Bolshevik Ali Bayramov |
| Mikoyan | Yeghegnadzor | 1935-1957 | Caucasus | after Anastas Mikoyan |
| Ordzhonikidzeabad | Vahdat | 1936-1992 | Central Asia | after Ordzhonikidze |
| Stalinabad | Dushanbe | 1929-1961 | Central Asia | after Stalin |
| Leninabad | Khujand | 1936-1991 | Central Asia | after Lenin |
| Frunze | Bishkek | 1926-1991 | Central Asia | after Mikhail Frunze |
| Tselinograd | Astana | 1961-1992 | Central Asia | Virgin Lands city |
| Dzhambul | Taraz | 1938-1993 | Central Asia | after poet Dzhambul Dzhabayev |
