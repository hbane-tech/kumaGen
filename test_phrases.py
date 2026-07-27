"""
test_phrases.py
Suite de tests exhaustive — toutes les phrases testées + matrice être/avoir complète.
Sources : sessions de débogage 2026-05-20 → 2026-06-15 + documents de référence.

Format : (phrase_fr, traduction_bambara_attendue, categorie)
Usage  : python test_phrases.py          # liste
         python test_phrases.py --run    # avec moteur
"""

TEST_CASES = [

    # ════════════════════════════════════════════════════════════════════
    # I. ÊTRE — IDENTIFICATION ÉQUATIVE
    # ════════════════════════════════════════════════════════════════════
    ("Je suis enseignant", "N yé karamɔgɔ yé", "equative_sing"),
    ("il est professeur", "a yé porofesɛri yé", "equative_sing"),
    ("Musa est un chasseur", "Musa yé donso yé", "equative_sing"),
    ("je suis étudiant", "n yé kalandenba yé", "equative_sing"),
    ("tu es mon ami", "i yé n teri yé", "equative_sing"),
    ("elle est médecin", "a yé dɔkɔtɔrɔ yé", "equative_sing"),
    ("nous sommes des paysans", "ánw yé wúlakɔnɔmɔgɔw yé", "equative_plur"),
    ("Ils sont étudiants", "ùw yé kàlandenbaw yé", "equative_plur"),
    ("vous êtes des enseignants", "aw yé karamɔgɔw yé", "equative_plur"),

    # Équatif négatif
    ("il n'est pas professeur", "a tɛ porofesɛri yé", "equative_neg"),
    ("il n'est pas un professeur", "a tɛ porofesɛri yé", "equative_neg"),
    ("il n'est pas un étudiant", "a tɛ kàlandenba yé", "equative_neg"),
    ("je ne suis pas enseignant", "n tɛ karamɔgɔ yé", "equative_neg"),
    ("Il n'est rien", "a tɛ foyi yé", "equative_neg_nothing"),
    ("ce n'est pas vrai", "o tɛ tiɲɛ yé", "equative_neg"),

    # Équatif passé
    ("il était professeur", "a tùn yé porofesɛri yé", "equative_past"),
    ("j'étais étudiant", "n tùn yé kalandenba yé", "equative_past"),
    ("il n'était pas étudiant", "a tùn tɛ kalandenba yé", "equative_past_neg"),

    # ════════════════════════════════════════════════════════════════════
    # II. ÊTRE — LOCALISATION
    # ════════════════════════════════════════════════════════════════════
    ("Je suis en route", "n bɛ síraden la", "locative"),
    ("Musa est au village", "Musa bɛ dùgu la", "locative"),
    ("Les enfants sont en route", "dɔgɔw bɛ síraden la", "locative_plur"),
    ("Ils sont aux portes", "u bɛ bóndaw la", "locative_plur"),
    ("je suis à la maison", "n bɛ so", "locative"),
    ("nous sommes au marché", "anw bɛ súgu la", "locative_plur"),
    ("il est au champ", "a bɛ foro la", "locative"),
    ("elle est à l'école", "a bɛ lakɔli la", "locative"),
    ("il n'est pas à la maison", "a tɛ so", "locative_neg"),
    ("je ne suis pas au marché", "n tɛ súgu la", "locative_neg"),

    # ════════════════════════════════════════════════════════════════════
    # III. ÊTRE — QUALIFICATION
    # ════════════════════════════════════════════════════════════════════
    ("je suis belle", "n ka cɛ̀ɲi", "qualitative"),
    ("le cheval est rapide", "sò ka téli", "qualitative"),
    ("La maison est loin", "só ka póroo", "qualitative"),
    ("Les maisons sont loin", "sów ka póroo", "qualitative_plur"),
    ("l'eau est chaude", "jí ká gàn", "qualitative"),
    ("la route est longue", "síraden ka jàn", "qualitative"),
    ("tu n'es pas bien", "i mán ɲi", "qualitative_neg"),
    ("la maison n'est pas grande", "só man bòn", "qualitative_neg"),
    ("la route était longue", "síraden tùn ka jàn", "qualitative_past"),
    ("il n'était pas bien", "a tùn mán ɲi", "qualitative_past_neg"),

    # ════════════════════════════════════════════════════════════════════
    # IV. ÊTRE — ÉTAT STATIF
    # ════════════════════════════════════════════════════════════════════
    ("je suis sûre de ça", "ń jɛ́len dòn o la", "statif"),
    ("Les maisons sont sûres", "sów jɛ́len dòn", "statif_plur"),
    ("C'est cuit", "o mɔnna", "past_intransitive"),
    ("Les viandes sont cuites", "sɔgɔw mɔnna", "past_intransitive_plur"),
    ("il est parti", "o fáɲira", "past_intransitive"),
    ("elle est tombée", "a bìnna", "past_intransitive"),
    ("ils sont arrivés", "ùw sera", "past_intransitive_plur"),

    # ════════════════════════════════════════════════════════════════════
    # V. AVOIR — PASSÉ TRANSITIF
    # ════════════════════════════════════════════════════════════════════
    ("Musa a mangé le riz", "Musa yé màlo dún", "past_transitive"),
    ("J'ai acheté des pagnes", "n yé fìníw sàn", "past_transitive"),
    ("tu as vu l'homme", "i yé cɛ yé", "past_transitive"),
    ("elle a bu de l'eau", "a yé jí mìn", "past_transitive"),
    ("nous avons mangé le riz", "anw yé màlo dún", "past_transitive_plur"),
    ("ils ont construit la maison", "u ye só gòsi", "past_transitive_plur"),
    ("L'homme que tu as vu hier est parti", "i yé cɛ mìn yé kúnùn, o fáɲira", "relative_topic"),

    # ════════════════════════════════════════════════════════════════════
    # VI. AVOIR — POSSESSION MATÉRIELLE
    # ════════════════════════════════════════════════════════════════════
    ("J'ai de l'argent", "wárí bɛ n bóló", "noun_phrase_have_material"),
    ("il a une voiture", "wátiri bɛ a bóló", "noun_phrase_have_material"),
    ("Ils ont des voitures", "móbiliw bɛ ùw bóló", "noun_phrase_have_material_plur"),
    ("tu as un couteau", "mùru bɛ i bóló", "noun_phrase_have_material"),
    ("Quel âge as-tu ?", "i bɛ saan jùmɛn la ?", "noun_phrase_have_age"),
    ("elle n'a pas d'argent", "wárí tɛ a bóló", "noun_phrase_have_material_neg"),

    # ════════════════════════════════════════════════════════════════════
    # VII. AVOIR — POSSESSION ABSTRAITE
    # ════════════════════════════════════════════════════════════════════
    ("J'ai un frère", "bálimakɛ bɛ n fɛ", "noun_phrase_have_abstract"),
    ("J'ai une sœur", "bálimamuso bɛ n fɛ", "noun_phrase_have_abstract"),
    ("J'ai des frères et sœurs", "bálimakɛw ni bálimamusow bɛ n fɛ", "noun_phrase_have_abstract_plur"),
    ("J'aime manger", "n bɛ dúnli kànuya", "volitif"),
    ("il a de la chance", "gàrijɛgɛ bɛ a fɛ", "noun_phrase_have_abstract"),
    ("elle n'a pas de frère", "bálimakɛ tɛ a fɛ", "noun_phrase_have_abstract_neg"),

    # ════════════════════════════════════════════════════════════════════
    # VIII. EXISTENCE PURE
    # ════════════════════════════════════════════════════════════════════
    ("Il y'a du pain", "búuru bɛ", "existential_absolute"),
    ("il y a un problème", "báasi bɛ", "existential_absolute"),
    ("il y a du travail", "baara bɛ", "existential_absolute"),
    ("il n'y a pas de pain", "búuru tɛ", "existential_absolute_neg"),
    ("il n'y a personne dans le village", "mɔgɔ tɛ dugu kɔnɔ", "existential_localized_neg"),

    # ════════════════════════════════════════════════════════════════════
    # IX. EXISTENCE LOCALISÉE
    # ════════════════════════════════════════════════════════════════════
    ("Il y a de l'eau dans la bouteille", "jí bɛ bútèli kɔnɔ", "existential_localized"),
    ("Il y a des gens ici", "mɔgɔw bɛ yàn", "existential_localized"),
    ("Il n'y a personne ici", "mɔgɔw tɛ yàn", "existential_localized_neg"),
    ("il y a des enfants dans la maison", "dɔgɔw bɛ so kɔnɔ", "existential_localized"),
    ("il n'y a pas d'eau dans la bouteille", "jí tɛ bútèli kɔnɔ", "existential_localized_neg"),

    # ════════════════════════════════════════════════════════════════════
    # X. PRÉSENTATIF / IDENTIFICATOIRE
    # ════════════════════════════════════════════════════════════════════
    ("C'est moi", "n dòn", "identificatory"),
    ("C'est moi Hawa", "n de Hawa yé", "identificatory_appos"),
    ("Ce n'est pas moi", "n tɛ", "identificatory_neg"),
    ("c'est Musa", "Musa dòn", "presentative"),
    ("Ce sont mes frères et sœurs", "ò yé n bálimakɛw ni n bálimamusow yé", "presentative_plur"),
    ("c'est lui", "ale dòn", "identificatory"),
    ("ce n'est pas lui", "ale tɛ", "identificatory_neg"),
    ("c'est nous", "anw dòn", "identificatory"),

    # ════════════════════════════════════════════════════════════════════
    # XI. DÉICTIQUE
    # ════════════════════════════════════════════════════════════════════
    ("Voilà la voiture", "wátiri félé", "deictique"),
    ("Voilà les voitures", "Wátiriw félé", "deictique_plur"),
    ("Voilà Musa", "Musa félé", "deictique"),
    ("Voilà l'eau", "jí félé", "deictique"),

    # ════════════════════════════════════════════════════════════════════
    # XII. INTERROGATIVES OUI/NON
    # ════════════════════════════════════════════════════════════════════
    ("Parles-tu bambara ?", "í bɛ bámanankan fɔ́ wà ?", "interrogative"),
    ("Allez-vous au marché ?", "aw bɛ táa dɔ́gɔ la wà ?", "interrogative_motion"),
    ("Est-ce que ça va vraiment bien ?", "Yala nin bɛ táa yɛ̀rɛ kóɲuman wà ?", "interrogative_question_marker"),
    ("Est-ce que tes frères et sœurs sont ici ?", "Yala i bálimakɛw ni i balimamusow bɛ yàn wà ?", "interrogative_question_marker"),
    ("Tu veux du poisson ou bien tu veux de la viande ?", "i bɛ jɛ́gɛ ŋàniya wàlima i bɛ sògo ŋàniya ?", "interrogative_alternative"),
    ("Tu veux des arachides ou bien tu veux des patates douces ?", "i bɛ tìgaw ŋàniya wàlima i bɛ wósow ŋàniya ?", "interrogative_alternative"),
    ("il mange ?", "a bɛ dúnli kɛ wà ?", "interrogative"),
    ("tu as de l'argent ?", "wárí bɛ i bóló wà ?", "interrogative"),
    ("elle est à la maison ?", "a bɛ so wà ?", "interrogative"),
    ("peut-il manger de la viande ?", "a bɛ se ka sògo dún wà ?", "interrogative_modal_xcomp"),

    # ════════════════════════════════════════════════════════════════════
    # XIII. QUESTIONS DE CONTENU
    # ════════════════════════════════════════════════════════════════════
    ("Qui fait la bagarre ?", "jɔn bɛ bàlawu kɛ́ ?", "content_question_who"),
    ("Qui aimez-vous ?", "aw bɛ jɔn kànu ?", "content_question_who_inversion"),
    ("Qui aiment-ils ?", "u bɛ jɔn kànu ?", "content_question_who_inversion"),
    ("Tu manges quoi ?", "i bɛ mún dún ?", "content_question_what"),
    ("Tu veux quoi ?", "i bɛ mún ŋàniya ?", "content_question_what"),
    ("il mange quoi ?", "a bɛ mún dún ?", "content_question_what"),
    ("Où allez-vous ?", "aw bɛ táa mín ?", "content_question_where"),
    ("où est Musa ?", "Musa bɛ mín ?", "content_question_where"),
    ("Quand viens-tu ?", "i bɛ nà túma jùmɛn ?", "content_question_when"),
    ("quand pars-tu ?", "i bɛ fáɲi túma jùmɛn ?", "content_question_when"),
    ("Pourquoi pleures-tu ?", "i bɛ kàsi mún kósɔn ?", "content_question_why"),
    ("pourquoi il mange ?", "a bɛ dumuni kɛ mún kósɔn ?", "content_question_why"),
    ("Comment t'appelles-tu ?", "í bɛ wéle cógo dǐ ?", "content_question_how"),
    ("Comment cela se fait-il ?", "ó bɛ kɛ́ cógo dǐ ?", "content_question_how"),
    ("Quelle femme ?", "mùso jùmɛn ?", "content_question_which_noun"),
    ("Tu veux quelle maison ?", "í bɛ só jùmɛn ŋàniya ?", "content_question_which"),
    ("Ça coûte combien ?", "o bɛ jóli bɔ́ ?", "content_question_how_much"),
    ("tu as combien d'enfants ?", "dénw jóli bɛ í fɛ ?", "content_question_how_much"),

    # ════════════════════════════════════════════════════════════════════
    # XIV. SIMPLE PRÉSENT
    # ════════════════════════════════════════════════════════════════════
    ("je mange du riz", "n bɛ màlo dún", "simple"),
    ("tu bois de l'eau", "i bɛ jí mìn", "simple"),
    ("il parle bambara", "a bɛ bámanankan fɔ́", "simple"),
    ("elle chante", "a bɛ donkili dá", "simple"),
    ("nous travaillons", "anw bɛ báara la", "simple_plur"),
    ("ils mangent du riz", "ùw bɛ màlo dún", "simple_plur"),
    ("je ne mange pas", "n tɛ dún", "simple_neg"),
    ("il ne parle pas bambara", "a tɛ bámanankan fɔ́", "simple_neg"),

    # Volitif
    ("je veux manger", "n bɛ ŋàniya ka dúnli kɛ", "volitif"),
    ("il veut partir", "a bɛ ŋàniya ka fáɲi", "volitif"),
    ("je ne veux pas manger", "n tɛ ŋàniya ka dúnli kɛ", "volitif_neg"),

    # Progressif
    ("je suis en train de manger", "ń bɛ́ kà dúnli kɛ", "progressif"),
    ("il est en train de parler", "a bɛ́ kà kúma", "progressif"),

    # Futur
    ("je vais manger", "ń bɛ na dúnli kɛ", "futur"),
    ("il va partir", "a bɛ na fáɲi", "futur"),
    ("je n'irai pas", "n tɛ na táa", "futur_neg"),

    # Passé négatif
    ("je n'ai pas mangé", "ń ma dúnli kɛ", "past_neg"),
    ("il n'a pas parlé", "a ma kúma", "past_neg"),

    # Habitude
    ("il mangeait du riz", "a tùn bɛ màlo dún", "habitude"),
    ("je travaillais", "ń tùn bɛ báara la", "habitude"),

    # ════════════════════════════════════════════════════════════════════
    # XV. VERBE SÉRIEL / COMPLEXE
    # ════════════════════════════════════════════════════════════════════
    ("Je te donne de l'argent pour cuisiner et manger", "n bɛ wárí di i ma walasa ka tóbili kɛ́ ani ka dúnli kɛ", "simple_purposive_coordinated"),

    ("Je veux manger avec mon mari et ma fille qui est malade", "ń bɛ ŋàniya ka dúnli kɛ ni n fúrucɛ ni n dénmuso yé mìn jànkarotɔ dòn", "complex_comitative_relative"),

    ("il donne le livre à l'enfant", "a bɛ gáfe dí dén ma", "verb_serial_dative"),

    # ════════════════════════════════════════════════════════════════════
    # XVI. PRIVATIF
    # ════════════════════════════════════════════════════════════════════
    ("sans moi", "n kɔ", "privative_pron"),
    ("sans moi, tu ne pourras pas partir", "n kɔ, i tɛ na se ka fáɲi", "privative_clause"),
    ("Ce légume est sans cuisson", "nin lègimu in yé tóbibali yé", "privative_noun"),
    ("il est parti sans avertir", "a fáɲira ka sɔrɔ a ma sàrali kɛ", "privative_verb"),
    ("sans eau", "jítan", "privative_noun"),
    ("sans argent", "wárítan", "privative_noun"),

    # ════════════════════════════════════════════════════════════════════
    # XVII. RELATIF / TOPIC
    # ════════════════════════════════════════════════════════════════════

    ("Une femme qui a eu un enfant ne peut pas abandonner son enfant", "mùso mìn ye dén sɔrɔ, o tɛ sé ka a dén bìla", "relative_topic_neg"),

    ("Les djihadistes montrent clairement qu'ils sont les maîtres du jeu", "bànbaganciw bɛ jìrali kɛ fɛ́rɛtɛtɛ ko u yé túlon tìgiw yé", "equative_relative"),

    # ════════════════════════════════════════════════════════════════════
    # XVIII. COMITATIVE / RÉCIPROQUE
    # ════════════════════════════════════════════════════════════════════
    ("je suis avec mon mari", "n ni n cɛ dòn", "comitative"),
    ("Je suis avec la fille du frère de mon ami", "n ni n terikɛ balimakɛ denmuso dòn", "comitative_genitive"),
    ("je mange avec mon ami", "ń bɛ dúnli kɛ ni n téri yé", "comitative_action"),
    ("nous nous aimons", "ánw bɛ ɲɔgɔn fɛ́", "reciprocal"),
    ("ils se battent", "ù bɛ kɛlɛ kɛ", "reciprocal"),
    ("nous nous parlons", "ánw bɛ kúma ɲɔgɔn fɛ", "reciprocal"),

    # ════════════════════════════════════════════════════════════════════
    # XIX. IMPÉRATIF / PROHIBITIF
    # ════════════════════════════════════════════════════════════════════
    ("mange !", "dumuni kɛ", "imperative"),
    ("pars !", "fáɲi", "imperative"),
    ("viens !", "nà", "imperative"),
    ("parle bambara !", "bámanankan fɔ́", "imperative"),
    ("ne mange pas !", "kàna dumuni kɛ", "prohibitive"),
    ("ne pars pas !", "kàna fáɲi", "prohibitive"),
    ("ne parle pas !", "kàna kúma", "prohibitive"),

    # ════════════════════════════════════════════════════════════════════
    # XX. APPARTENANCE / SYNTAGME NOMINAL
    # ════════════════════════════════════════════════════════════════════
    ("ce sac est pour ma fille", "nin bɔ̀rɛ in yé n dénmuso ta yé", "ownership"),
    ("Ce sac est pour moi", "nin bɔ̀rɛ in yé n ta yé", "ownership_pron"),
    ("la maison de mon ami", "n terikɛ ka so", "noun_phrase_alienable"),
    ("la participation citoyenne et démocratique", "jàmaden jɔ̀yɔrɔ ani demokaratiki", "noun_phrase_coord"),
    ("La décision finale de la constitution du Mali", "Mali dácogo làtigɛ lában", "noun_phrase_genitive"),
    ("le champ du vieux", "kɔ̀rɔ ka fòro", "noun_phrase_genitive"),
    ("la mère de l'enfant", "dén ba", "noun_phrase_genitive"),
    ("la maison de Musa", "Musa ka so", "noun_phrase_genitive"),

    # ════════════════════════════════════════════════════════════════════
    # XXI. INFINITIF
    # ════════════════════════════════════════════════════════════════════
    ("Promouvoir le bambara et les autres langues nationales", "ka bámanankan ni násɔnali kán wɛ́rɛ bárabɔ", "infinitive"),
    ("manger du riz", "ka màlo dún", "infinitive"),
    ("parler bambara", "ka bámanankan fɔ́", "infinitive"),
    ("ne pas manger", "kàna dúnli kɛ", "infinitive_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXII. PHRASES LONGUES / COMPLEXES
    # ════════════════════════════════════════════════════════════════════
    ("Promouvoir le bambara et les autres langues nationales, inclusion sociale, meilleure circulation de l'information", "ka bámanankan ni Nasiyonali kán wɛ́rɛ bárabɔ, sendonli sosiyaliman, kùnnafoni ka jɛnsɛncogo fìsaman", "infinitive_list"),

    ("Mais la situation reste très incertaine au Mali, qui s'enfonce toujours plus dans le chaos", "nka kísa bɛ tó siga la kojugu Mali la, mìn bɛ a yɛrɛ tíntin túgun kùnmafili kɔnɔ", "complex_relative"),

    ("Il existe un frein politique lié aux logiques de pouvoir des élites qui veulent maintenir la masse populaire à l'écart de la gouvernance", "fɛrɛn pólitiki sìrilen bɛ ŋàaraw ka fànga sàriyaw la min bɛ ŋàniya ka jàma popilɛri lámìnɛ màra nkànfulori ma", "existential_nominal_complex"),

    ("Chercheur associé à l'Institut français des relations internationales, Thierry Vircoulon revient pour 20 Minutes sur la situation explosive au Mali", "ɲíninikɛla jɛ̀len Institut faransɛ cɛ́sira ɛntɛrinasiyɔnali la, Thierry Vircoulon bɛ sègin 20 mínitiw ye kísa bàlawumaman kan Mali la", "complex_noun_phrase"),

    ("L'université de Kumamoto recrute actuellement des participants pour le Programme d'apprentissage coopératif japonais, ainsi que des étudiants japonais pour la soutenir", "Kumamoto iniwɛrisite bɛ jɔ̀yɔrɔtigiw cɛ̀ta hálisà dègeli kóperatifu zapɔnɛ pòrogaramu ye, ka fara kàlandenbaw zapɔnɛ kan walasa ka a bánban", "complex_purposive"),

    # ════════════════════════════════════════════════════════════════════
    # XXIII. CORRECTIFS SESSION 2026-06
    # ════════════════════════════════════════════════════════════════════

    # Locatif ADV pur avec être (ici/là ROOT, pas existentiel)
    ("les gens sont ici", "mɔ̀gɔw bɛ yàn", "locative_adv"),
    ("il est là", "a bɛ yèn", "locative_adv"),

    # Numéraux cardinaux — sujet et objet
    ("deux hommes sont partis", "mɔ̀gɔw fila fáɲira", "nummod_subject"),
    ("trois enfants mangent du riz", "dénw saba bɛ iri dún", "nummod_subject"),
    ("cinq femmes sont ici", "númanfɛlakaw duuru bɛ yàn", "nummod_locative"),
    ("j'ai acheté deux livres", "n yé kìtabuw fila sàn", "nummod_object"),

    # Verbe intransitif ACTION en INTRANS_SC (báara = travailler)
    ("je suis en train de travailler", "ń bɛ́ kà báara kɛ", "intrans_action_progressif"),
    ("je ne travaille pas", "ń tɛ báara la", "intrans_action_present_neg"),
    ("j'ai travaillé", "ń ye báara kɛ", "intrans_action_passe_pos"),
    ("je n'ai pas travaillé", "ń ma báara kɛ", "intrans_action_passe_neg"),
    ("je ne travaillais pas", "n tùn tɛ báara la", "intrans_action_hab_neg"),
    ("je travaillerai", "ń tùn tɛ báara la", "intrans_action_futur"),

    # Subordonnant temporel (quand → tuma min antéposé)
    ("quand il vient", "túma mín a bɛ nà", "temporal_subordinator"),

    # Discours rapporté + comparatif (plus ADJ que X → ka ADJ ka tɛmɛ X kan)
    ("les bambara disent que la raison de la venue de quelqu'un est plus importante que soi-même", "bambaraw bɛ fɔ́ ko mɔ̀gɔ dɔ jɔ̀kun ka kùnba ka tɛmɛ a yɛrɛ kan", "reported_comparative"),

    # ════════════════════════════════════════════════════════════════════
    # XXIV. CORRECTIFS SESSION 2026-06-12
    # ════════════════════════════════════════════════════════════════════

    # Optatif / subjonctif : que + Mood=Sub → S ka (O) V
    ("que tu viennes", "í ka nà", "optatif"),
    ("que Moussa mange", "Moussa ka dumuni kɛ", "optatif"),
    ("que Dieu t'aide", "Ala ka i dɛmɛ", "optatif_coi"),

    # Venir de + lieu (bɔra) — PROPN sans 'la', NOUN avec 'la'
    ("Je viens de Bamako", "ń bɛ bɔ Bamako la", "venir_de_propn"),
    ("Je viens de l'école", "ń bɛ bɔ làkɔli la", "venir_de_noun"),

    # Venir de + verbe (passé récent) → bɔra ka + verbe avec transitivité
    ("Je viens de manger", "ń bɔra ka dúnli kɛ", "venir_de_verbe"),
    ("Il vient de partir", "a bɔra ka fáɲi", "venir_de_verbe"),

    # Verbe coordonné avec oblique locatif (oblique avant la clause coordonnée)
    ("elle alla au village et demande des infos", "a táara wa a ye ɛnfo ɲíni dùgu la", "conj_avec_locatif"),

    # Verbe sériel + verbe coordonné sur le xcomp
    ("elle alla trouver et demande des informations", "a táara ka yé wa a bɛ kùnnafoni ɲíni", "verb_serial_conj"),

    # Comitatif + verbe de mouvement ABSOLU (pas de 'li kɛ', pas de doublon ni/yé)
    ("il est venu avec moi", "a nàra ni n yé", "comitative_motion"),
    ("elle travaille avec lui", "a bɛ báara la ni a yé", "comitative_travail"),

    # Comitatif + adjectif sur le nom comitatif
    ("Il est venu avec une dentition complète", "a nàra ni ɲín dafalen yé", "comitative_adj"),

    # ════════════════════════════════════════════════════════════════════
    # XXV. SYNTAGME NOMINAL TEMPOREL — CORRECTIFS 2026-06-13
    # ════════════════════════════════════════════════════════════════════

    # Préposition temporelle + quantifier + nom → kabini préfixé, nom pluriel, dɔw postposé
    ("Depuis quelques mois", "kabini sélidenninkalow dɔw", "temporal_np_depuis"),
    ("depuis quelques jours", "kabini tilew dɔw", "temporal_np_depuis"),
    ("depuis quelques années", "kabini sanw dɔw", "temporal_np_depuis"),

    # ════════════════════════════════════════════════════════════════════
    # XXVI. SUBORDONNÉE COMPLÉTIVE — ko + clause (ccomp)
    # ════════════════════════════════════════════════════════════════════

    # Verbe cognitif savoir/dɔn — ko + clause équative
    ("sais-tu que je suis un enfant ?", "í bɛ dɔ́n ko ń yé dén yé wà ?", "ccomp_interrogative"),
    ("je sais que tu es mon ami", "ń bɛ dɔ́n ko í yé n téri yé", "ccomp_savoir"),
    ("ils savent que nous sommes ici", "u bɛ dɔ́n ko ánw bɛ yàn", "ccomp_savoir_locatif"),

    # Verbe de parole dire — présent : S ko [ccomp] (sans TAM ni verbe dire)
    ("il dit qu'il mange", "a ko a bɛ dúnli kɛ", "ccomp_dire"),
    ("ils disent que la route est longue", "u ko síraden ka jàn", "ccomp_dire_qualitative"),
    # Passé : S TAM VERBE ko [ccomp] (chemin normal conservé)
    ("elle a dit que Musa est parti", "a yé fɔ ko Musa fáɲira", "ccomp_dire_passe"),

    # ════════════════════════════════════════════════════════════════════
    # XXVII. MODAL — VARIATIONS SUR "POUVOIR / SE KA"
    # ════════════════════════════════════════════════════════════════════

    ("nous pouvons partir", "anw bɛ se ka táa", "modal_pouvoir"),
    ("il ne peut pas manger", "a tɛ sé ka dúnli kɛ", "modal_pouvoir_neg"),
    ("elle peut venir", "a bɛ se ka nà", "modal_pouvoir"),
    ("tu ne peux pas partir", "i tɛ se ka fáɲi", "modal_pouvoir_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXVIII. PASSÉ INTRANSITIF NÉGATIF
    # ════════════════════════════════════════════════════════════════════

    ("ils ne sont pas arrivés", "ùw ma se", "past_intransitive_neg"),
    ("il n'est pas tombé", "a ma bìn", "past_intransitive_neg"),
    ("elle n'est pas venue", "a ma nà", "past_intransitive_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXIX. HABITUDE NÉGATIVE
    # ════════════════════════════════════════════════════════════════════

    ("elle ne mangeait pas", "a tùn tɛ dúnli kɛ", "habitude_neg"),
    ("nous ne parlions pas bambara", "anw tùn tɛ bámanankan fɔ́", "habitude_neg_plur"),
    ("il n'avait pas mangé", "a tùn ma dúnli kɛ", "passe_anterieur_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXX. RÈGLES SPÉCIALISÉES (Rules 1-9)
    # ════════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════════
    # RÈGLES SPÉCIALISÉES (Rules 1-9)
    # ════════════════════════════════════════════════════════════════════

    # Rule 1: Conditional + question → ends with 'dun ?' not 'wà ?'
    ("Si j'ai du courage, saurait-on ?", "ní dùsukolo bɛ n fɛ, a bɛ́nà dɔ́n wà ?", "rule1_conditional_question"),

    # Rule 2: Prohibitive + object → 'kàna [object] V'
    ("Ne mange pas le riz", "kàna iri dún", "rule2_prohibitive_object"),

    # Rule 3: Temporal + passé simple avoir + possessive
    ("quand il eut ton appel", "túma mín a yé i ka wéle sɔrɔ", "rule3_temporal_avoir_possessive"),

    # Rule 4: Temporal + passive passé simple
    ("Quand cela fut fait", "túma mín ó tùn kɛ́ra", "rule4_temporal_passive"),

    # Rule 5: Fixed phrase 'Ainsi donc'
    ("Ainsi donc", "ola sa", "rule5_fixed_phrase"),

    # Rule 6: ne...que restrictive → 'S TAM foyi yé ni ATTR tɛ'
    ("tu ne serais qu'un pleutre", "í tɛ́nà kɛ dɔwɛrɛ yé ni jítɔ tɛ", "rule6_restrictive"),

    # Rule 7: Qu'est-ce que = mún S TAM V ka O V_ACT [obliques] (expletive: no S)
    ("Qu'est-ce qu'il pourrait t'arriver là-bas ?", "mún bɛ́nà sé ka nà i ma yèn ?", "rule7_quest_ce_que"),
    ("Qu'est-ce qu'il peut faire ?", "a bɛ sé ka mún kɛ́ ?", "rule7_quest_ce_que_real_subject"),

    # Rule 8: Complex relative with reflexive + correct clause splitting
    ("Toi qui prends l'ennemi vivant", "e mìn bɛ júgu ɲɛ́nama tà", "rule8_relative_splitting"),

    # Rule 9: valoir la peine idiom (follows general SOV rule)
    ("cela vaut la peine de prendre un fusil", "ó bɛ sɛ̀gɛn bɔ́ ka màrifa tà", "rule9_valoir_peine_sov"),

    # Additional variants for better coverage
    ("tu ne serais que jaloux", "í tɛ́nà kɛ dɔwɛrɛ yé ni kèle tɛ", "rule6_restrictive_adj"),

    # ════════════════════════════════════════════════════════════════════
    # XXXI. RÉFLEXIF ABSOLU — arbre de décision (sessions 2026-06-15)
    # ════════════════════════════════════════════════════════════════════
    # Structure : S TAM refl_pron [yɛrɛ] V
    # Cat. Actif (volontaire, yɛrɛ=False par défaut)
    ("il s'est lavé", "a yé a kò", "refl_actif_passe"),
    ("elle se lave", "a bɛ a kò", "refl_actif_present"),

    # Cat. Actif + emphase explicite 'lui même' → yɛrɛ=True
    ("il s'est lavé lui même", "a ye a yɛrɛ kò", "refl_actif_emphase"),

    # Cat. Accidentel (involontaire, yɛrɛ=True)
    ("il s'est blessé", "a ye a yɛrɛ jógin", "refl_accidentel_passe"),

    # Posture (is_refl_Subjective, Cat. Actif)
    ("il s'est assis", "a ye a sìgi", "refl_posture_passe"),

    # Idiomatique + xcomp locatif → S yé refl_pron yɛrɛ V_MAIN V_ACT la
    ("il s'est mis à pleurer", "a yé a yɛrɛ bìla kàsi la", "refl_idiom_locatif"),

    # ════════════════════════════════════════════════════════════════════
    # XXXII. CORRECTIONS SESSION 2026-06-17
    # ════════════════════════════════════════════════════════════════════

    # Pronom objet indirect (iobj) à la 1re personne
    ("Il me parle", "a bɛ kúma n fɛ", "simple_iobj_present"),
    ("elle me donne un livre", "a bɛ kìtabu di n ma", "verb_serial_dative_me"),

    # Question de contenu + modal + xcomp (pouvoir faire ?)
    ("Qu'est-ce qu'il pourrait faire ?", "a bɛ́nà sé ka mún kɛ́ ?", "content_question_modal_xcomp"),

    # ════════════════════════════════════════════════════════════════════
    # XXXIII. AVOIR AU FUTUR — POSSESSION FUTURE
    # ════════════════════════════════════════════════════════════════════

    ("Tu auras une voiture", "í bɛ́nà wátiri sɔrɔ", "future_have_material"),
    ("Elle aura de l'argent", "a bɛ́nà wári sɔrɔ", "future_have_material"),
    ("J'aurai un frère", "n bɛ́nà bálimakɛ sɔrɔ", "future_have_abstract"),
    ("Il aura du succès", "a bɛ́nà táɲɛw sɔrɔ", "future_have_abstract"),
    ("Nous aurons des enfants", "anw bɛ́nà dénw sɔrɔ", "future_have_abstract_plur"),
    ("ils n'auront pas de voiture", "u tɛ́nà wátiri sɔrɔ", "future_have_material_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXXIV. AVOIR EU — ACQUISITION PASSÉE (yé ... sɔrɔ)
    # ════════════════════════════════════════════════════════════════════

    ("j'ai eu un mari", "n yé fúrucɛ sɔrɔ", "past_transitive_avoir_eu"),
    ("elle a eu des enfants", "a yé dénw sɔrɔ", "past_transitive_avoir_eu_plur"),
    ("tu as eu de la chance", "í yé gàrijɛgɛ sɔrɔ", "past_transitive_avoir_eu"),
    ("il a eu un problème", "a yé báasi sɔrɔ", "past_transitive_avoir_eu"),
    ("nous avons eu des difficultés", "ánw yé lújuraw sɔrɔ", "past_transitive_avoir_eu_plur"),

    # ════════════════════════════════════════════════════════════════════
    # XXXV. POSSESSION PASSÉE — HABITUDE (tùn bɛ ... bóló / fɛ)
    # ════════════════════════════════════════════════════════════════════

    ("j'avais une voiture", "móbili tùn bɛ n bóló", "past_have_material"),
    ("il avait de l'argent", "wárí tùn bɛ a bóló", "past_have_material"),
    ("elle n'avait pas de frère", "bálimakɛ tùn tɛ a fɛ", "past_have_abstract_neg"),
    ("nous avions un champ", "foro tùn bɛ anw bóló", "past_have_material_plur"),
    ("tu avais de la chance", "tère tùn bɛ i fɛ", "past_have_abstract"),

    # ════════════════════════════════════════════════════════════════════
    # XXXVI. OPTATIF ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("Que Dieu vous bénisse", "Ala ka aw bárika", "optatif_plur"),
    ("Qu'il vienne", "a ka nà", "optatif"),
    ("Qu'ils partent", "ùw ka táa", "optatif_plur"),
    ("Que la paix règne", "hɛ́ɛrɛ ka kɛ", "optatif_abs"),
    ("Que tu réussisses", "í ka ɲɛ̀", "optatif"),
    ("Que nous mangions ensemble", "ánw ka dúnli kɛ ɲɔgɔn fɛ", "optatif_plur"),

    # ════════════════════════════════════════════════════════════════════
    # XXXVII. COORDINATION VERBALE ÉTENDUE
    # ════════════════════════════════════════════════════════════════════

    # S TAM V1 wa S TAM V2 (présent)
    ("il mange et boit", "a bɛ dúnli kɛ wa a bɛ mìn", "conj_verb_present"),
    ("elle chante et danse", "a bɛ donkilidá wa a bɛ dɔ̀n kɛ́", "conj_verb_present"),
    # Passé coordonné
    ("il est venu et a mangé", "a nàra wa a ye dumuni kɛ", "conj_verb_passe"),

    # ════════════════════════════════════════════════════════════════════
    # XXXVIII. MODAL ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("tu peux manger", "í bɛ sé ka dúnli kɛ", "modal_pouvoir"),
    ("nous ne pouvons pas venir", "ánw tɛ sé ka nà", "modal_pouvoir_neg"),
    ("elles peuvent travailler", "u bɛ sé ka báara kɛ", "modal_pouvoir_plur"),
    ("il pouvait partir", "a tùn bɛ se ka fáɲi", "modal_pouvoir_past"),
    ("peut-elle partir ?", "a bɛ se ka fáɲi wà ?", "modal_pouvoir_interrogative"),
    ("il ne pouvait pas manger", "a tùn tɛ sé ka dúnli kɛ", "modal_pouvoir_past_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXXIX. IOBJ / DATIF ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("il me donne de l'argent", "a bɛ wári dí ń ma", "verb_serial_dative"),
    ("elle lui parle", "a bɛ kúma ale fɛ", "simple_iobj"),
    ("nous leur donnons du riz", "ánw bɛ màlo dí u ma", "verb_serial_dative_plur"),
    ("je te donne un livre", "ń bɛ gáfe kelen dí í ma", "verb_serial_dative"),
    ("il nous parle", "a bɛ kúma anw fɛ", "simple_iobj_plur"),

    # ════════════════════════════════════════════════════════════════════
    # XL. RÉFLEXIF ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("il se lève", "a bɛ wúli", "refl_posture_present"),
    ("il s'est levé", "a wúlila", "refl_posture_passe"),
    ("elle s'est trompée", "a ŋànamula", "refl_accidentel_passe"),
    ("il s'est mis à travailler", "a yé a yɛrɛ bìla báara la", "refl_idiom_locatif"),
    ("elle se regarde", "a bɛ a yɛrɛ fílɛ", "refl_actif_present"),
    ("nous nous préparons", "ánw bɛ ánw yɛrɛ labɛn", "refl_actif_present_plur"),
    ("elle se réveille", "a bɛ kúnun", "refl_posture_present"),

    # ════════════════════════════════════════════════════════════════════
    # XLI. IMPÉRATIF ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("viens ici !", "nà yàn", "imperative_locative"),
    ("ne fais pas ça !", "kàna o kɛ", "prohibitive_dem"),
    ("aide-moi !", "n dɛ̀mɛ", "imperative_iobj"),
    ("ne pleure pas !", "kàna kàsi", "prohibitive"),
    ("mange vite !", "dúnli kɛ joona", "imperative_adv"),
    ("ne bois pas l'eau !", "kàna jí mìn", "prohibitive_object"),
    ("ne mange pas la viande !", "kàna sògo dún", "prohibitive_object"),

    # ════════════════════════════════════════════════════════════════════
    # XLII. INTERROGATIVE ÉTENDUE
    # ════════════════════════════════════════════════════════════════════

    ("Est-ce qu'il mange ?", "Yala a bɛ dúnli kɛ wà ?", "interrogative_est_ce_que"),
    ("Est-ce qu'elle travaille ?", "Yala a bɛ báara la wà ?", "interrogative_est_ce_que"),
    ("a-t-il mangé ?", "a ye dúnli kɛ wà ?", "interrogative_passe"),
    ("Travailles-tu ?", "í bɛ báara la wà ?", "interrogative"),
    ("est-il venu ?", "a nàra wà ?", "interrogative_past_intrans"),
    ("Est-ce qu'ils sont arrivés ?", "Yala ùw sera wà ?", "interrogative_est_ce_que_passe"),

    # ════════════════════════════════════════════════════════════════════
    # XLIII. QUESTIONS DE CONTENU ÉTENDUES
    # ════════════════════════════════════════════════════════════════════

    ("Qui est-il ?", "a yé jɔn yé ?", "content_question_who_equative"),
    ("Qu'est-ce que tu fais ?", "í bɛ mún kɛ́ ?", "content_question_what"),
    ("Quand est-il parti ?", "a fáɲira túma jùmɛn ?", "content_question_when_passe"),
    ("Pourquoi ne viens-tu pas ?", "í tɛ nà mún kósɔn ?", "content_question_why_neg"),
    ("Combien coûte ce livre ?", "nin gáfe in bɛ jóli bɔ́ ?", "content_question_how_much"),
    ("Qui a mangé ?", "jɔn ye dúnli kɛ ?", "content_question_who_passe"),

    # ════════════════════════════════════════════════════════════════════
    # XLIV. RELATIVES ÉTENDUES
    # ════════════════════════════════════════════════════════════════════

    ("La femme qui mange", "mùso mìn bɛ dúnli kɛ", "relative_subject"),
    ("Le livre que j'ai acheté", "n ye gáfe mìn sàn", "relative_object"),
    ("L'homme qui est parti", "mɔ̀gɔ mìn fáɲira", "relative_subject_past"),
    ("L'enfant qui pleure", "dén mìn bɛ kàsi", "relative_subject_present"),
    ("La maison que nous avons construite", "ánw ye só mìn gòsi", "relative_object_past"),

    # ════════════════════════════════════════════════════════════════════
    # XLV. CCOMP ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("tu sais qu'il est parti ?", "i bɛ dɔ́n ko a fáɲira wà ?", "ccomp_interrogative_passe"),
    # dire au présent → S ko [ccomp] (TAM + verbe dire supprimés)
    ("elle dit qu'elle partira", "a ko a bɛ na táa", "ccomp_dire_futur"),
    ("nous savons qu'il est là", "ánw bɛ dɔ́n ko a bɛ yèn", "ccomp_locatif"),
    ("il dit qu'il ne mange pas", "a ko a tɛ dúnli kɛ", "ccomp_dire_neg"),
    ("je sais que tu es là", "ń bɛ dɔ́n ko í bɛ yèn", "ccomp_locatif"),
    ("ils disent qu'il est venu", "ùw ko a nàra", "ccomp_dire_passe_ccomp"),
    # dire au passé → S TAM VERBE ko [ccomp] (chemin normal conservé)
    ("il a dit qu'il mangeait", "a yé fɔ ko a tùn bɛ dúnli kɛ", "ccomp_dire_passe_root"),
    ("ils ont dit qu'il était parti", "ùw yé fɔ ko a tùn fáɲira", "ccomp_dire_passe_root_plur"),

    # ════════════════════════════════════════════════════════════════════
    # XLVI. VENIR DE ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("Ils viennent du marché", "u bɛ bɔ dɔ́gɔ la", "venir_de_noun_plur"),
    ("Elle vient de la mosquée", "a bɛ bɔ mìsiri la", "venir_de_noun"),
    ("Nous venons de Bamako", "anw bɛ bɔ Bamako", "venir_de_propn_plur"),
    ("Elle vient de travailler", "a bɔra ka báara kɛ", "venir_de_verbe"),
    ("tu viens d'où ?", "i bɛ bɔ mín ?", "venir_de_interrogative"),

    # ════════════════════════════════════════════════════════════════════
    # XLVII. COMITATIVE ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("il est venu avec ses enfants", "a nàra ni a dénw yé", "comitative_plur_noun"),
    ("je mange avec ma famille", "ń bɛ dúnli kɛ ni n ka dénbaya yé", "comitative_action"),
    ("il est parti avec ses amis", "a fáɲira ni a teriw yé", "comitative_passe"),
    ("elle travaille avec son mari", "a bɛ báara la ni a cɛ yé", "comitative_travail"),

    # ════════════════════════════════════════════════════════════════════
    # XLVIII. RÈGLES SPÉCIALISÉES — COUVERTURE ÉTENDUE
    # ════════════════════════════════════════════════════════════════════

    # Rule 2 : prohibitif + objet (variantes)
    ("Ne bois pas le lait", "kàna nɔnɔ mìn", "rule2_prohibitive_object"),
    ("Ne prends pas ça", "kàna ò tà", "rule2_prohibitive_dem"),

    # Rule 6 : ne...que (variantes)
    ("tu n'es qu'un enfant", "í tɛ dɔwɛrɛ yé ni dén tɛ", "rule6_restrictive"),
    ("il n'est qu'un lâche", "a tɛ dɔwɛrɛ yé ni jítɔ tɛ", "rule6_restrictive"),
    ("elle n'est que belle", "a tɛ dɔwɛrɛ yé ni ɲuman tɛ", "rule6_restrictive_adj"),

    # Rule 7 : Qu'est-ce que / contenu expletif (variantes)
    ("Qu'est-ce qu'il a fait ?", "a ye mún kɛ́ ?", "content_question_past"),
    ("Qu'est-ce que vous voulez ?", "áw bɛ mún ŋàniya ?", "rule7_quest_ce_que_plur"),

    # Rule 8 : relative + construction complexe (variantes)
    ("L'homme que tu vois est mon ami", "í bɛ mɔ̀gɔ mìn yé , o yé n téri yé", "rule8_relative_topic"),

    # Rule 9 : valoir la peine (variantes)
    ("cela ne vaut pas la peine", "ó tɛ sɛ̀gɛn bɔ́", "rule9_valoir_peine_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XLIX. EXPERIENCER STATE CONSTRUCTION — avoir + état subjectif
    # Bambara : STATE bɛ SUBJ la  /  SUBJ_POSS CORPS bɛ SUBJ PAIN
    # ════════════════════════════════════════════════════════════════════

    ("J'ai faim", "kɔ́ngɔ bɛ n la", "experiencer_faim"),
    ("J'ai soif", "jáabi bɛ n la", "experiencer_soif"),
    ("il a peur", "a jàpapalen dòn", "experiencer_peur"),
    ("elle a honte", "maloya bɛ a la", "experiencer_honte"),
    ("j'ai sommeil", "sùnɔgɔ bɛ n la", "experiencer_sommeil"),
    ("nous avons froid", "nɛ́nɛ bɛ ánw la", "experiencer_froid"),
    ("tu as chaud", "fùnteni bɛ í la", "experiencer_chaud"),
    ("je n'ai pas faim", "kɔngɔ tɛ n la", "experiencer_faim_neg"),
    ("il n'a pas peur", "a jàpapalen tɛ", "experiencer_peur_neg"),

    # Cas douleur + partie du corps
    ("j'ai mal à la tête", "n kùn bɛ n dimi", "experiencer_pain_tete"),
    ("elle a mal au ventre", "a kɔnɔ bɛ a dimi", "experiencer_pain_ventre"),
    ("il a mal aux pieds", "a sènw bɛ a dimi", "experiencer_pain_pied"),

    # ════════════════════════════════════════════════════════════════════
    # L. FRÉQUENCE ET DISTRIBUTION TEMPORELLE
    # Bambara : distributif sɛbɛ / tɛmɛnen kɔrɔ / lɔgɔ kelen kelen…
    # ════════════════════════════════════════════════════════════════════

    ("tous les jours", "dón bɛ́ɛ", "freq_tous_les_jours"),
    ("chaque jour", "dón ó dón", "freq_chaque_jour"),
    ("chaque mois", "kálo ó kálo", "freq_chaque_mois"),
    ("chaque semaine", "dɔ́gɔfurancɛ ò dɔ́gɔfurancɛ", "freq_chaque_semaine"),
    ("chaque année", "saan ó saan", "freq_chaque_annee"),
    ("tous les matins", "sɔgɔma bɛ́ɛ", "freq_tous_les_matins"),
    ("tous les soirs", "wúlaw bɛ́ɛ", "freq_tous_les_soirs"),
    ("il vient tous les jours", "a bɛ nà dón bɛ́ɛ", "freq_vient_tous_les_jours"),
    ("elle mange chaque matin", "a bɛ dúnli kɛ sɔ̀gɔmà ò sɔ̀gɔmà", "freq_mange_chaque_matin"),
    ("nous travaillons chaque semaine", "ánw bɛ báara kɛ dɔ́gɔfurancɛ ò dɔ́gɔfurancɛ", "freq_travaille_chaque_semaine"),

    # ════════════════════════════════════════════════════════════════════
    # M. PHRASES IMPERSONNELLES
    # Météorologiques (phénomène bɛ na), obligation (a ka kan ka V),
    # condition météo (a bɛ X la)
    # ════════════════════════════════════════════════════════════════════

    # Météorologiques
    ("il pleut", "sán bɛ́ nà", "impersonnel_meteo"),
    ("il neige", "nɛzi bɛ́ nà", "impersonnel_meteo"),
    ("il fait chaud", "fùnteni bɛ", "impersonnel_meteo"),
    ("il fait froid", "nɛ́nɛ bɛ", "impersonnel_meteo"),
    ("il fait nuit", "sú kòra", "impersonnel_meteo"),

    # Obligation / nécessité (il faut / il ne faut pas)
    ("il faut manger", "a ka kan ka dúnli kɛ", "impersonnel_falloir"),
    ("il faut partir", "a ka kan ka táa", "impersonnel_falloir"),
    ("il ne faut pas manger", "a man kan ka dún", "impersonnel_falloir_neg"),
    ("il ne faut pas partir", "a man kan ka táa", "impersonnel_falloir_neg"),

    # Devoir / obligation personnelle
    ("il doit partir", "a ka kan ka táa", "impersonnel_devoir"),
]


# ── POS / DEPENDENCY TREE EXTRACTION ───────────────────────────────────────────
# French : notre propre parseur spaCy (pipeline/spacy_parser.py), déjà utilisé
# pour l'analyse en amont de la traduction — donne pos/dep/head_index réels.
# Bambara : arbre de RÉFÉRENCE construit depuis tree_meta (clause_type +
# slots S/TAM/O/V connus avec certitude, cf. eval/bambara_ud_reference.py),
# suivant le schéma documenté par Aplonova & Tyers 2017/2018 et vérifié
# contre le treebank réel UD_Bambara. PAS le modèle UDPipe tiers pour les
# clause_type couverts : vérifié sur "n yé kàramɔgɔkɛ yé" (équatif réel
# Kuma), le modèle UDPipe inverse les rôles NOUN/VERB (copule 'yé' taggée
# NOUN, nom 'kàramɔgɔkɛ' taggé VERB+root) — cf. décision 2026-07-08.
# Fallback UDPipe conservé uniquement pour les clause_type non couverts par
# bambara_ud_reference (construction non vérifiée contre le papier/corpus).
from eval.bambara_udpipe import parse_bambara, pos_tree_string, dep_tree_string
from eval.bambara_ud_reference import build_reference_ud_tree


def get_french_dep_tree(phrase, tagger=None):
    """Parse le français avec notre propre spaCy parser → liste de tokens
    {form, upos, head, deprel} (mêmes clés que parse_bambara pour comparaison)."""
    if not tagger:
        return []
    try:
        toks = tagger.parse(phrase)
        return [{
            'id':     i + 1,
            'form':   t.get('surface', ''),
            'upos':   t.get('pos', 'X'),
            'head':   (t.get('head_index', 0) + 1) if t.get('head_index', i) != i else 0,
            'deprel': t.get('dep', 'dep'),
        } for i, t in enumerate(toks)]
    except Exception:
        return []


def get_french_pos_tree(phrase, tagger=None):
    """Extract POS sequence from French phrase for syntactic tree comparison."""
    return pos_tree_string(get_french_dep_tree(phrase, tagger))


def get_bambara_dep_tree(phrase_fr, bambara_result, engine=None, tagger=None, tree_meta=None):
    """Arbre de référence (build_reference_ud_tree) si le clause_type est
    couvert ; sinon repli sur le modèle UDPipe tiers (moins fiable, mais
    mieux que rien pour les constructions non encore vérifiées)."""
    if tree_meta:
        ref = build_reference_ud_tree(tree_meta, bambara_result)
        if ref:
            return ref
    return parse_bambara(bambara_result)


def get_bambara_pos_tree(phrase_fr, bambara_result, engine=None, tagger=None, tree_meta=None):
    """Extract UPOS sequence from the Bambara reference/UDPipe tree."""
    return pos_tree_string(get_bambara_dep_tree(phrase_fr, bambara_result, engine, tagger, tree_meta))


# ── CAMEMBERT SEMANTIC SIMILARITY (per-word, per-system) ──────────────────────
# NOTE (bug fix): the previous implementation back-translated Bambara → French
# for ALL THREE systems via `googletrans`, then compared whole-sentence
# CamemBERT embeddings. Two problems made every score collapse to the same
# value regardless of which system was actually better:
#   1. `googletrans` (4.0.0rc1) is broken against modern httpcore —
#      `Translator()` raises AttributeError at call time.
#   2. Even if it worked, `'bm'` (Bambara) isn't in googletrans' hardcoded
#      LANGUAGES table at all.
# Every call silently failed (bare `except Exception: return ''`), so
# every back-translation was '' → embedded as a blank placeholder → all three
# systems were being scored against the same degenerate blank text, not their
# actual output.
#
# Fixed by (a) using a back-translator that actually supports Bambara per
# system, and (b) comparing at the WORD level, aligned per source word rather
# than pooling a whole sentence into one vector:
#   - Kuma-MT : each retrieved KG token already carries its own French gloss
#     (`sens_fr`, the `fr` field of the KG Sense node) tied directly to the
#     source word it translated — no back-translation or alignment needed,
#     just cosine(embed(source_word), embed(kg_gloss)) per retrieved token.
#   - Google/NLLB : no such per-word correspondence exists (they return one
#     opaque Bambara sentence), so each Bambara word of their output is
#     back-translated INDIVIDUALLY, POS-tagged, and matched to the original
#     source word sharing that POS tag (falling back to the best candidate
#     overall if none share the POS — single-word POS tagging out of context
#     is unreliable, so a hard POS-only match would silently zero out
#     mismatches that are actually fine translations).
#   - Google back-translation via deep_translator's GoogleTranslator (unlike
#     googletrans, it actually supports 'bm'); NLLB back-translation via the
#     NLLB model itself (bam_Latn → fra_Latn) — same model family as the
#     baseline.
_camembert_model = None
_nllb_bt_model = None
_nllb_bt_tokenizer = None
_word_embed_cache = {}     # word -> vector, avoids re-encoding repeats
_backtrans_cache = {'google': {}, 'nllb': {}}  # bm piece -> fr word, avoids repeat calls


def _load_camembert():
    global _camembert_model
    if _camembert_model is None:
        from sentence_transformers import SentenceTransformer
        print("  [CamemBERT] Chargement du modèle dangvantuan/sentence-camembert-large …")
        _camembert_model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    return _camembert_model


def _cam_embed(word: str):
    """Cached CamemBERT embedding for a single word/short phrase."""
    import numpy as np
    word = (word or '').strip()
    if not word:
        return None
    if word not in _word_embed_cache:
        model = _load_camembert()
        _word_embed_cache[word] = model.encode(
            word, convert_to_numpy=True, normalize_embeddings=True)
    return _word_embed_cache[word]


def _cam_cosine(w1: str, w2: str):
    import numpy as np
    v1, v2 = _cam_embed(w1), _cam_embed(w2)
    if v1 is None or v2 is None:
        return None
    return float(np.dot(v1, v2))


def _cam_embed_contextual(words: list) -> list:
    """Embeddings CamemBERT CONTEXTUELS pour une liste de mots, encodés
    ENSEMBLE comme une seule séquence (une seule passe du modèle) — chaque
    vecteur reflète donc ses voisins, contrairement à _cam_embed() qui
    encode un mot seul, sans aucun contexte de phrase.

    Utilisé exclusivement par BERTScore (bertscore_prf) : BERTScore analyse
    la phrase entière (contextuel), CamemBERT (score_kuma_word_level /
    score_backtrans_word_level) reste mot-par-mot isolé (_cam_embed).

    Récupère les embeddings de sous-mots (subword) via
    output_value='token_embeddings' (une passe complète du modèle sur toute
    la séquence), puis les regroupe par mot d'origine via word_ids() du
    tokenizer (moyenne des sous-mots d'un même mot) — les tokens spéciaux
    (<s>, </s>) sont exclus (word_id=None)."""
    import torch
    words = [w.strip() for w in (words or []) if w and w.strip()]
    if not words:
        return []
    text = ' '.join(words)
    model = _load_camembert()
    tok_emb = model.encode([text], output_value='token_embeddings')[0]   # (n_subtok, dim)
    word_ids = model.tokenizer(text).word_ids()

    buckets = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        buckets.setdefault(wid, []).append(tok_emb[i])

    vecs = []
    for wid in range(len(words)):
        pieces = buckets.get(wid)
        if not pieces:
            continue
        v = torch.stack(pieces).mean(dim=0)
        v = v / v.norm()
        vecs.append(v.detach().cpu().numpy())
    return vecs


def backtranslate_google_bm_to_fr(bm_text: str) -> str:
    """Back-translate a Bambara word/piece → French via Google Translate (deep_translator)."""
    if not bm_text:
        return ''
    if bm_text in _backtrans_cache['google']:
        return _backtrans_cache['google'][bm_text]
    try:
        from deep_translator import GoogleTranslator
        out = GoogleTranslator(source='bm', target='fr').translate(bm_text) or ''
    except Exception as e:
        print(f"       [Google back-trad] échec sur {bm_text!r}: {e}")
        out = ''
    _backtrans_cache['google'][bm_text] = out
    return out


def _load_nllb_backtranslator():
    global _nllb_bt_model, _nllb_bt_tokenizer
    if _nllb_bt_model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _name = 'facebook/nllb-200-distilled-600M'
        print(f"  [NLLB back-trad] Chargement de {_name} …")
        _nllb_bt_tokenizer = AutoTokenizer.from_pretrained(_name, src_lang='bam_Latn')
        _nllb_bt_model = AutoModelForSeq2SeqLM.from_pretrained(_name)
    return _nllb_bt_model, _nllb_bt_tokenizer


def backtranslate_nllb_bm_to_fr(bm_text: str) -> str:
    """Back-translate a Bambara word/piece → French via the NLLB model itself (bam_Latn → fra_Latn)."""
    if not bm_text:
        return ''
    if bm_text in _backtrans_cache['nllb']:
        return _backtrans_cache['nllb'][bm_text]
    try:
        model, tokenizer = _load_nllb_backtranslator()
        inputs = tokenizer(bm_text, return_tensors='pt', truncation=True)
        gen = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids('fra_Latn'),
            max_new_tokens=20,
        )
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"       [NLLB back-trad] échec sur {bm_text!r}: {e}")
        out = ''
    _backtrans_cache['nllb'][bm_text] = out
    return out


def score_kuma_word_level(kuma_tokens: list) -> float:
    """
    CamemBERT (mot-à-mot, ISOLÉ) : chaque mot comparé UN-À-UN à son propre
    mot source, chacun embeddé À PART, sans contexte de phrase (_cam_embed).
    C'est le contraire de BERTScore (bertscore_prf/kuma_bertscore_prf) qui
    encode la phrase ENTIÈRE en une passe (contextuel) et fait un alignement
    glouton sur toute la phrase — décision 2026-07-09 : les deux métriques
    doivent rester distinctes par construction, pas juste par le nom.

    Couvre TOUS les tokens traités par Kuma (pas seulement ceux glosés par
    le KG) : un mot-outil sans glose utilise son propre lemme comme mot de
    comparaison (cosine(lemme, lemme) = 1.0, trivial mais cohérent avec la
    même convention de couverture complète déjà adoptée pour BERTScore)
    plutôt que d'être ignoré.
    """
    sims = []
    for t in (kuma_tokens or []):
        if t.get('pos') in ('PUNCT', 'SYM'):
            continue
        src_word = (t.get('lemma') or t.get('surface') or '').strip()
        if not src_word:
            continue
        gloss = (t.get('sens_fr') or '').strip()
        hyp_word = gloss.split('.')[0].split(',')[0].strip() if gloss else src_word
        if not hyp_word:
            continue
        sim = _cam_cosine(src_word, hyp_word)
        if sim is not None:
            sims.append(sim)
    return sum(sims) / len(sims) if sims else None


def score_backtrans_word_level(phrase_fr: str, bm_sentence: str,
                                backtranslate_word_fn, tagger) -> float:
    """
    CamemBERT (mot-à-mot, ISOLÉ) pour Google/NLLB : pas d'alignement par
    token possible sur une sortie phrase entière, donc chaque mot bambara
    est back-traduit individuellement, POS-taggé, et apparié au mot de la
    phrase source qui partage ce POS (meilleur cosinus parmi les candidats
    de même POS ; repli sur le meilleur score global si aucun match POS,
    le tagging d'un mot isolé hors contexte étant peu fiable). Chaque
    embedding reste isolé (_cam_embed) — contrairement à
    backtrans_bertscore_prf qui encode toute la phrase reconstituée en une
    passe (contextuel).
    """
    if not bm_sentence or not phrase_fr or not tagger:
        return None
    try:
        orig_tagged = tagger.parse(phrase_fr)
    except Exception:
        return None
    orig_words = [(t.get('lemma') or t.get('surface', ''), t.get('pos', 'X'))
                  for t in orig_tagged if t.get('pos') not in ('PUNCT', 'SYM')]
    orig_words = [(w.strip(), p) for w, p in orig_words if w.strip()]
    if not orig_words:
        return None

    bm_pieces = [w.strip('.,?;:!') for w in bm_sentence.split()]
    bm_pieces = [w for w in bm_pieces if w]
    if not bm_pieces:
        return None

    piece_info = []  # (fr_word, pos)
    for piece in bm_pieces:
        fr_word = backtranslate_word_fn(piece)
        if not fr_word:
            continue
        try:
            _tagged = tagger.parse(fr_word)
            pos = _tagged[0].get('pos', 'X') if _tagged else 'X'
        except Exception:
            pos = 'X'
        piece_info.append((fr_word.strip(), pos))
    if not piece_info:
        return None

    sims = []
    for orig_word, pos in orig_words:
        same_pos = [fw for fw, p in piece_info if p == pos]
        candidates = same_pos or [fw for fw, _ in piece_info]
        scored = [(_cam_cosine(orig_word, fw), fw) for fw in candidates]
        scored = [(s, fw) for s, fw in scored if s is not None]
        if scored:
            sims.append(max(scored)[0])
    return sum(sims) / len(sims) if sims else None


def bertscore_prf(hyp_words: list, ref_words: list):
    """
    BERTScore CONTEXTUEL : hyp_words et ref_words sont chacun encodés
    ENSEMBLE comme une seule séquence (une seule passe du modèle par côté,
    via _cam_embed_contextual) — chaque vecteur mot reflète donc ses
    voisins, contrairement à CamemBERT (score_kuma_word_level /
    score_backtrans_word_level) qui embeddent chaque mot isolément
    (_cam_embed, aucun contexte de phrase). C'est la vraie distinction entre
    les deux métriques (décision 2026-07-09), pas juste un algo différent
    sur les mêmes vecteurs.
      - Kuma   : hyp = glose KG des tokens retrouvés (concaténées en pseudo-
                 phrase), ref = phrase FR source réelle
      - Google/NLLB : hyp = pièces back-traduites mot-à-mot (concaténées),
                 ref = phrase FR source réelle

    P = avg over hyp words of max cosine to any ref word
    R = avg over ref words of max cosine to any hyp word
    F1 = harmonic mean(P, R)
    Returns (P, R, F1) or (None, None, None) if either side is empty.
    """
    hyp_vecs = _cam_embed_contextual(hyp_words)
    ref_vecs = _cam_embed_contextual(ref_words)
    if not hyp_vecs or not ref_vecs:
        return None, None, None
    import numpy as np
    sims = np.array([[float(np.dot(h, r)) for r in ref_vecs] for h in hyp_vecs])
    P = float(sims.max(axis=1).mean())
    R = float(sims.max(axis=0).mean())
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    return P, R, F1


def kuma_bertscore_prf(kuma_tokens: list, phrase_fr: str, tagger):
    """
    Kuma P/R/F1 : hyp = TOUS les tokens traités par Kuma (pas seulement ceux
    avec une glose KG), ref = phrase FR source.

    Un token avec sens_fr (mot de contenu retrouvé dans le KG) utilise sa
    glose ; un mot-outil (pronom, être/avoir auxiliaire, article...) n'a pas
    de glose KG mais Kuma l'a quand même traité — on utilise alors son propre
    lemme comme mot d'hypothèse, plutôt que de l'omettre entièrement (ce qui
    faisait chuter artificiellement le rappel : la référence contient TOUS
    les mots de la phrase, l'hypothèse doit pouvoir les couvrir tous aussi).

    Les verbes sont comparés à l'infinitif des deux côtés : le lemme spaCy
    d'un verbe conjugué EST son infinitif par convention ("dit"→"dire"), et
    la glose KG d'un sens Verb suit la même convention lexicographique
    (entrée de dictionnaire) — aucune normalisation supplémentaire requise
    tant qu'on utilise lemma/gloss et jamais surface pour un verbe.
    """
    hyp_words = []
    for t in (kuma_tokens or []):
        if t.get('pos') in ('PUNCT', 'SYM'):
            continue
        gloss = (t.get('sens_fr') or '').strip()
        if gloss:
            hyp_words.append(gloss.split('.')[0].split(',')[0].strip())
        else:
            lemma = (t.get('lemma') or t.get('surface') or '').strip()
            if lemma:
                hyp_words.append(lemma)
    if not hyp_words or not phrase_fr or not tagger:
        return None, None, None
    try:
        ref_tagged = tagger.parse(phrase_fr)
    except Exception:
        return None, None, None
    ref_words = [t.get('lemma') or t.get('surface', '') for t in ref_tagged
                 if t.get('pos') not in ('PUNCT', 'SYM')]
    ref_words = [w.strip() for w in ref_words if w.strip()]
    return bertscore_prf(hyp_words, ref_words)


def backtrans_bertscore_prf(phrase_fr: str, bm_sentence: str,
                             backtranslate_word_fn, tagger):
    """Google/NLLB P/R/F1 : hyp = pièces back-traduites mot-à-mot, ref = phrase FR source."""
    if not bm_sentence or not phrase_fr or not tagger:
        return None, None, None
    try:
        ref_tagged = tagger.parse(phrase_fr)
    except Exception:
        return None, None, None
    ref_words = [t.get('lemma') or t.get('surface', '') for t in ref_tagged
                 if t.get('pos') not in ('PUNCT', 'SYM')]
    ref_words = [w.strip() for w in ref_words if w.strip()]
    if not ref_words:
        return None, None, None

    bm_pieces = [w.strip('.,?;:!') for w in bm_sentence.split()]
    bm_pieces = [w for w in bm_pieces if w]
    if not bm_pieces:
        return None, None, None
    hyp_words = [backtranslate_word_fn(p) for p in bm_pieces]
    hyp_words = [w.strip() for w in hyp_words if w and w.strip()]
    return bertscore_prf(hyp_words, ref_words)


class _NoOpStemmer:
    """Pas de stemmer pour METEOR en bambara — le PorterStemmer par défaut
    de nltk applique des règles morphologiques ANGLAISES ; les appliquer à
    des formes bambara produirait des troncatures arbitraires, pas de vraies
    racines partagées. Le stage synonymes de meteor_score() cherche aussi
    dans WordNet (anglais) — inerte pour le bambara (aucun synset), donc
    METEOR se réduit ici à : match exact unigramme + pénalité de
    fragmentation (ordre des mots) — toujours une mesure valide et
    directement comparable entre les 3 systèmes, contrairement à
    BERTScore/CamemBERT qui exigent une back-traduction bambara→français
    faute de modèle d'embedding bambara."""
    def stem(self, word):
        return word


_meteor_wordnet_ready = False


def meteor_bm(hyp: str, ref: str):
    """METEOR bambara-vs-bambara DIRECT (pas de back-traduction requise,
    contrairement à BERTScore/CamemBERT) : hyp = sortie du système (Kuma/
    Google/NLLB), ref = 'bambara_attendu' (traduction de référence humaine
    déjà présente dans TEST_CASES). Retourne None si hyp ou ref est vide."""
    if not hyp or not ref:
        return None
    global _meteor_wordnet_ready
    try:
        from nltk.translate.meteor_score import meteor_score
        if not _meteor_wordnet_ready:
            import nltk
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            _meteor_wordnet_ready = True
        hyp_toks = [w.strip('.,?;:!').lower() for w in hyp.split()]
        ref_toks = [w.strip('.,?;:!').lower() for w in ref.split()]
        hyp_toks = [w for w in hyp_toks if w]
        ref_toks = [w for w in ref_toks if w]
        if not hyp_toks or not ref_toks:
            return None
        return meteor_score([ref_toks], hyp_toks, stemmer=_NoOpStemmer())
    except Exception as e:
        print(f"       [METEOR] échec: {e}")
        return None


# ── RUNNER ────────────────────────────────────────────────────────────────────
def run_tests(translate_fn=None, baseline_cache: dict = None, no_camembert: bool = False):
    """
    baseline_cache : dict avec clés 'google' et 'nllb'
                     → {phrase_fr: traduction_bm}
                     Chargé depuis baseline_translations.json si disponible.
    """
    import csv, datetime, json, os
    # Charger le cache baseline si non fourni explicitement
    if baseline_cache is None:
        _cache_path = os.path.join(os.path.dirname(__file__), 'baseline_translations.json')
        if os.path.exists(_cache_path):
            with open(_cache_path, encoding='utf-8') as _f:
                baseline_cache = json.load(_f)
    _google_cache = (baseline_cache or {}).get('google', {})
    _nllb_cache   = (baseline_cache or {}).get('nllb', {})

    total  = len(TEST_CASES)
    passed = 0
    failed = []
    results = []

    cats = {}
    for _, _, c in TEST_CASES:
        cats.setdefault(c, 0)
        cats[c] += 1

    print(f"\n{'='*72}")
    print(f"  SUITE DE TESTS BAMBARA — {total} phrases / {len(cats)} catégories")
    print(f"{'='*72}\n")

    # Initialize tagger if we're translating
    tagger = None
    if translate_fn:
        try:
            from pipeline.spacy_parser import SpacyParser
            from kg.neo4j_client import Neo4jClient
            db = Neo4jClient()
            tagger = SpacyParser(db)
        except:
            tagger = None

    for i, (phrase, expected, category) in enumerate(TEST_CASES, 1):
        print(f"[{i:02d}/{total}] {category}")
        print(f"  FR  : {phrase}")

        google_bm = _google_cache.get(phrase, '')
        nllb_bm   = _nllb_cache.get(phrase, '')

        if translate_fn:
            try:
                _tr = translate_fn(phrase)
                result      = _tr['bambara'].strip()
                kuma_tokens = _tr.get('tokens', [])
                tree_meta   = _tr.get('tree', {})
                ok = result == expected.strip()

                # Extract POS + dependency trees (FR: spaCy parser, BM: arbre
                # de référence build_reference_ud_tree, repli UDPipe sinon)
                fr_dep = get_french_dep_tree(phrase, tagger)
                bm_dep = get_bambara_dep_tree(phrase, result, engine=None,
                                              tagger=tagger, tree_meta=tree_meta)
                fr_pos = pos_tree_string(fr_dep)
                bm_pos = pos_tree_string(bm_dep)
                fr_dep_str = dep_tree_string(fr_dep)
                bm_dep_str = dep_tree_string(bm_dep)

                if ok:
                    passed += 1
                    print(f"    : {result}")
                    results.append((i, category, phrase, expected, result, 'PASS',
                                    fr_pos, bm_pos, fr_dep_str, bm_dep_str,
                                    google_bm, nllb_bm, kuma_tokens))
                else:
                    failed.append((i, category, phrase, expected, result))
                    print(f"    : {result}")
                    results.append((i, category, phrase, expected, result, 'FAIL',
                                    fr_pos, bm_pos, fr_dep_str, bm_dep_str,
                                    google_bm, nllb_bm, kuma_tokens))
            except Exception as e:
                err = f"ERREUR: {e}"
                failed.append((i, category, phrase, expected, err))
                print(f"    : {e}")
                results.append((i, category, phrase, expected, err, 'ERROR',
                                '', '', '', '', google_bm, nllb_bm, []))
        else:
            results.append((i, category, phrase, expected, '', '', '', '', '', '',
                            google_bm, nllb_bm, []))
        print()

    if translate_fn:
        print(f"{'='*72}")
        print(f"  RÉSULTATS : {passed}/{total} réussis  ({100*passed//total}%)")
        if failed:
            print(f"\n  ── ÉCHECS ({len(failed)}) ──")
            for n, cat, fr, att, got in failed:
                print(f"\n  [{n:02d}] [{cat}] {fr}")
                print(f"       ATT: {att}")
                print(f"       GOT: {got}")
        fail_cats = {}
        for n, cat, fr, att, got in failed:
            fail_cats.setdefault(cat, 0)
            fail_cats[cat] += 1
        if fail_cats:
            print(f"\n  ── PAR CATÉGORIE ──")
            for cat in sorted(fail_cats):
                print(f"   {cat}: {fail_cats[cat]}/{cats.get(cat,0)}")
        print(f"{'='*72}\n")
    else:
        print(f"{'='*72}")
        print(f"  {total} phrases — {len(cats)} catégories\n")
        print("  Catégories :")
        for cat in sorted(cats):
            print(f"    {cats[cat]:2d}  {cat}")
        print(f"\n  Pour tester : python test_phrases.py --run")
        print(f"{'='*72}\n")

    # ── Back-traduction PHRASE ENTIÈRE (Google/NLLB), pour lecture humaine ──
    # Distincte de la back-traduction MOT-À-MOT utilisée en interne par
    # CamemBERT/BERTScore (backtranslate_*_bm_to_fr appelée par morceau,
    # cf. score_backtrans_word_level/backtrans_bertscore_prf) : ici la
    # phrase bambara complète est back-traduite en une seule fois, pour
    # que l'expert humain puisse lire directement "ce que Google/NLLB a
    # compris" sans reconstituer un patchwork mot-à-mot.
    backtrans_google = [''] * len(results)
    backtrans_nllb   = [''] * len(results)

    if translate_fn:
        try:
            print("  [Back-trad phrase entière] Google/NLLB …")
            for r in results:
                _goog_bm, _nllb_bm = r[10], r[11]
                backtrans_google[r[0] - 1] = backtranslate_google_bm_to_fr(_goog_bm) if _goog_bm else ''
                backtrans_nllb[r[0] - 1]   = backtranslate_nllb_bm_to_fr(_nllb_bm) if _nllb_bm else ''
        except Exception as _e:
            print(f"  [Back-trad phrase entière] Erreur : {_e} — colonnes laissées vides")

    # ── CamemBERT semantic similarity, per word (batch) ───────────────
    cam_kuma   = [''] * len(results)
    cam_google = [''] * len(results)
    cam_nllb   = [''] * len(results)

    if translate_fn and not no_camembert:
        try:
            print("  [CamemBERT] Kuma   : comparaison isolée par token retrouvé dans le KG …")
            print("  [CamemBERT] Google : back-trad mot-à-mot (deep_translator) + alignement POS …")
            print("  [CamemBERT] NLLB   : back-trad mot-à-mot (modèle NLLB local) + alignement POS …")

            _sc_k, _sc_g, _sc_n = [], [], []
            for r in results:
                _phrase, _status = r[2], r[5]
                _goog_bm, _nllb_bm, _kuma_tokens = r[10], r[11], r[12]

                # CamemBERT (mot-à-mot, isolé) — distinct de BERTScore
                # (contextuel, glouton) : cf. décision 2026-07-09.
                _s_k = score_kuma_word_level(_kuma_tokens) if _status != 'ERROR' else None
                _s_g = score_backtrans_word_level(_phrase, _goog_bm, backtranslate_google_bm_to_fr, tagger)
                _s_n = score_backtrans_word_level(_phrase, _nllb_bm, backtranslate_nllb_bm_to_fr, tagger)

                if _s_k is not None: _sc_k.append(_s_k)
                if _s_g is not None: _sc_g.append(_s_g)
                if _s_n is not None: _sc_n.append(_s_n)

                cam_kuma[r[0] - 1]   = f'{_s_k:.3f}' if _s_k is not None else ''
                cam_google[r[0] - 1] = f'{_s_g:.3f}' if _s_g is not None else ''
                cam_nllb[r[0] - 1]   = f'{_s_n:.3f}' if _s_n is not None else ''

            avg_k = sum(_sc_k) / len(_sc_k) if _sc_k else 0.0
            avg_g = sum(_sc_g) / len(_sc_g) if _sc_g else 0.0
            avg_n = sum(_sc_n) / len(_sc_n) if _sc_n else 0.0
            print(f"\n  ── CamemBERT (mot-à-mot isolé, vs FR original) ──")
            print(f"  Kuma-MT : {avg_k:.3f}  (n={len(_sc_k)}, gloss KG par token retrouvé)")
            print(f"  Google  : {avg_g:.3f}  (n={len(_sc_g)}, back-trad + alignement POS)")
            print(f"  NLLB    : {avg_n:.3f}  (n={len(_sc_n)}, back-trad + alignement POS)\n")
        except Exception as _e:
            print(f"  [CamemBERT] Erreur : {_e} — colonnes laissées vides")

    # ── BERTScore contextuel P/R/F1 (batch) ────────────────────────────
    # Distinct de CamemBERT ci-dessus : chaque côté (hyp/ref) est encodé
    # EN UNE SEULE PASSE comme séquence complète (bertscore_prf via
    # _cam_embed_contextual), donc chaque vecteur mot reflète son contexte
    # de phrase — contrairement à CamemBERT qui embeddent chaque mot isolé.
    # Kuma calculé ICI (pas après-coup depuis le CSV) car kuma_tokens
    # (glose KG par token, nécessaire pour l'hypothèse Kuma) n'est
    # disponible qu'au moment du run — le CSV n'exporte que la chaîne
    # bambara finale, pas les tokens internes (décision 2026-07-20 : capturer
    # p_kuma/r_kuma/f1_kuma maintenant plutôt que de re-router par le CSV).
    bertscore_kuma   = [''] * len(results)
    bertscore_google = [''] * len(results)
    bertscore_nllb   = [''] * len(results)

    if translate_fn and not no_camembert:
        try:
            print("  [BERTScore] Kuma/Google/NLLB : encodage contextuel P/R/F1 …")
            _bs_k, _bs_g, _bs_n = [], [], []
            for r in results:
                _phrase, _status = r[2], r[5]
                _goog_bm, _nllb_bm, _kuma_tokens = r[10], r[11], r[12]

                pk, rk, f1k = kuma_bertscore_prf(_kuma_tokens, _phrase, tagger) \
                    if _status != 'ERROR' else (None, None, None)
                pg, rg, f1g = backtrans_bertscore_prf(_phrase, _goog_bm, backtranslate_google_bm_to_fr, tagger)
                pn, rn, f1n = backtrans_bertscore_prf(_phrase, _nllb_bm, backtranslate_nllb_bm_to_fr, tagger)

                if f1k is not None: _bs_k.append(f1k)
                if f1g is not None: _bs_g.append(f1g)
                if f1n is not None: _bs_n.append(f1n)

                bertscore_kuma[r[0] - 1]   = f'{pk:.3f}|{rk:.3f}|{f1k:.3f}' if f1k is not None else ''
                bertscore_google[r[0] - 1] = f'{pg:.3f}|{rg:.3f}|{f1g:.3f}' if f1g is not None else ''
                bertscore_nllb[r[0] - 1]   = f'{pn:.3f}|{rn:.3f}|{f1n:.3f}' if f1n is not None else ''

            avg_bk = sum(_bs_k) / len(_bs_k) if _bs_k else 0.0
            avg_bg = sum(_bs_g) / len(_bs_g) if _bs_g else 0.0
            avg_bn = sum(_bs_n) / len(_bs_n) if _bs_n else 0.0
            print(f"\n  ── BERTScore F1 (contextuel, vs FR original) ──")
            print(f"  Kuma-MT : {avg_bk:.3f}  (n={len(_bs_k)})")
            print(f"  Google  : {avg_bg:.3f}  (n={len(_bs_g)})")
            print(f"  NLLB    : {avg_bn:.3f}  (n={len(_bs_n)})\n")
        except Exception as _e:
            print(f"  [BERTScore] Erreur : {_e} — colonnes laissées vides")

    # ── METEOR, bambara-vs-bambara direct (batch) ──────────────────────
    # Contrairement à CamemBERT/BERTScore, pas de back-traduction requise :
    # les 3 systèmes sont comparés DIRECTEMENT à 'bambara_attendu' (déjà en
    # bambara dans TEST_CASES), donc les 3 scores sont mesurés sur exactement
    # la même référence — pas de biais introduit par la qualité variable
    # d'un back-traducteur tiers pour chaque système.
    meteor_kuma   = [''] * len(results)
    meteor_google = [''] * len(results)
    meteor_nllb   = [''] * len(results)

    if translate_fn:
        try:
            _mt_k, _mt_g, _mt_n = [], [], []
            for r in results:
                _expected, _status, _kuma_out = r[3], r[5], r[4]
                _goog_bm, _nllb_bm = r[10], r[11]

                _m_k = meteor_bm(_kuma_out, _expected) if _status != 'ERROR' else None
                _m_g = meteor_bm(_goog_bm, _expected)
                _m_n = meteor_bm(_nllb_bm, _expected)

                if _m_k is not None: _mt_k.append(_m_k)
                if _m_g is not None: _mt_g.append(_m_g)
                if _m_n is not None: _mt_n.append(_m_n)

                meteor_kuma[r[0] - 1]   = f'{_m_k:.3f}' if _m_k is not None else ''
                meteor_google[r[0] - 1] = f'{_m_g:.3f}' if _m_g is not None else ''
                meteor_nllb[r[0] - 1]   = f'{_m_n:.3f}' if _m_n is not None else ''

            avg_mk = sum(_mt_k) / len(_mt_k) if _mt_k else 0.0
            avg_mg = sum(_mt_g) / len(_mt_g) if _mt_g else 0.0
            avg_mn = sum(_mt_n) / len(_mt_n) if _mt_n else 0.0
            print(f"\n  ── METEOR (bambara vs bambara_attendu, direct) ──")
            print(f"  Kuma-MT : {avg_mk:.3f}  (n={len(_mt_k)})")
            print(f"  Google  : {avg_mg:.3f}  (n={len(_mt_g)})")
            print(f"  NLLB    : {avg_mn:.3f}  (n={len(_mt_n)})\n")
        except Exception as _e:
            print(f"  [METEOR] Erreur : {_e} — colonnes laissées vides")

    # ── Export CSV ─────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"resultats_bambara_{ts}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '#', 'categorie', 'phrase_fr',
            'bambara_attendu', 'bambara_obtenu', 'statut',
            'french_pos_tree', 'bambara_pos_tree',
            'french_dep_tree', 'bambara_dep_tree',
            'google_translate', 'nllb_translate',
            'google_backtrans_fr', 'nllb_backtrans_fr',
            'camembert_kuma', 'camembert_google', 'camembert_nllb',
            'meteor_kuma', 'meteor_google', 'meteor_nllb',
            'bertscore_prf_kuma', 'bertscore_prf_google', 'bertscore_prf_nllb',
        ])
        for row, btg, btn, ck, cg, cn, mk, mg, mn, bk, bg, bn in zip(
                results, backtrans_google, backtrans_nllb,
                cam_kuma, cam_google, cam_nllb,
                meteor_kuma, meteor_google, meteor_nllb,
                bertscore_kuma, bertscore_google, bertscore_nllb):
            writer.writerow(list(row[:12]) + [btg, btn, ck, cg, cn, mk, mg, mn, bk, bg, bn])  # row[12] = kuma_tokens (internal only, not a CSV cell)
    _has_baseline = bool(_google_cache or _nllb_cache)
    print(f"  CSV exporté : {csv_path}")
    print(f"  Colonnes : bambara_obtenu | google_translate | nllb_translate"
          + (" (cache baseline chargé)" if _has_baseline else " (cache baseline absent)"))
    if translate_fn:
        print(f"           | google_backtrans_fr | nllb_backtrans_fr  (phrase entière, BM→FR)")
    if translate_fn and not no_camembert:
        print(f"           | camembert_kuma | camembert_google | camembert_nllb")
    if translate_fn:
        print(f"           | meteor_kuma | meteor_google | meteor_nllb")
    if translate_fn and not no_camembert:
        print(f"           | bertscore_prf_kuma | bertscore_prf_google | bertscore_prf_nllb  (format 'P|R|F1')")
    print()
    return csv_path


if __name__ == '__main__':
    import sys
    _no_cam = '--no-camembert' in sys.argv
    if '--run' in sys.argv:
        try:
            sys.path.insert(0, '.')
            from pipeline.translation_engine import TranslationEngine
            from kg.neo4j_client import Neo4jClient
            db     = Neo4jClient()
            engine = TranslationEngine(db)
            run_tests(translate_fn=lambda s: engine.translate(s),
                      no_camembert=_no_cam)
        except ImportError as e:
            print(f"Moteur non trouvé ({e})\n")
            run_tests(no_camembert=_no_cam)
        except Exception as e:
            print(f"Erreur d'initialisation : {e}\n")
            run_tests(no_camembert=_no_cam)
    else:
        run_tests(no_camembert=_no_cam)