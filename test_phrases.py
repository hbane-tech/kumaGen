"""
test_phrases.py
Suite de tests exhaustive — toutes les phrases testées + matrice être/avoir complète.
Sources : sessions de débogage 2026-05-20 → 2026-05-21 + documents de référence.

Format : (phrase_fr, traduction_bambara_attendue, categorie)
Usage  : python test_phrases.py          # liste
         python test_phrases.py --run    # avec moteur
"""

TEST_CASES = [

    # ════════════════════════════════════════════════════════════════════
    # I. ÊTRE — IDENTIFICATION ÉQUATIVE
    # ════════════════════════════════════════════════════════════════════
    ("Je suis enseignant",                "n yé karamɔgɔ yé",              "equative_sing"),
    ("il est professeur",                 "a yé porofesɛri yé",             "equative_sing"),
    ("Musa est un chasseur",              "Musa yé donso yé",               "equative_sing"),
    ("je suis étudiant",                  "n yé kalandenba yé",             "equative_sing"),
    ("tu es mon ami",                     "i yé n terikɛ yé",               "equative_sing"),
    ("elle est médecin",                  "a yé dɔkɔtɔrɔ yé",              "equative_sing"),
    ("nous sommes des paysans",           "anw yé dugukɔlɔsibagaw yé",      "equative_plur"),
    ("Ils sont étudiants",                "ùw yé kalandenw yé",             "equative_plur"),
    ("vous êtes des enseignants",         "aw yé karamɔgɔw yé",             "equative_plur"),

    # Équatif négatif
    ("il n'est pas professeur",           "a tɛ porofesɛri yé",             "equative_neg"),
    ("il n'est pas un professeur",        "a tɛ porofesɛri yé",             "equative_neg"),
    ("il n'est pas un étudiant",          "a tɛ kàlandenba yé",             "equative_neg"),
    ("je ne suis pas enseignant",         "n tɛ karamɔgɔ yé",              "equative_neg"),
    ("Il n'est rien",                     "a tɛ foyi yé",                   "equative_neg_nothing"),
    ("ce n'est pas vrai",                 "o tɛ tiɲɛ yé",                   "equative_neg"),

    # Équatif passé
    ("il était professeur",               "a tùn yé porofesɛri yé",         "equative_past"),
    ("j'étais étudiant",                  "n tùn yé kalandenba yé",         "equative_past"),
    ("il n'était pas étudiant",           "a tùn tɛ kalandenba yé",         "equative_past_neg"),

    # ════════════════════════════════════════════════════════════════════
    # II. ÊTRE — LOCALISATION
    # ════════════════════════════════════════════════════════════════════
    ("Je suis en route",                  "n bɛ sìra la",                   "locative"),
    ("Musa est au village",               "Musa bɛ dugu kɔnɔ",              "locative"),
    ("Les enfants sont en route",         "dɔgɔw bɛ sìra la",               "locative_plur"),
    ("Ils sont aux portes",               "u bɛ bóndaw la",                 "locative_plur"),
    ("je suis à la maison",               "n bɛ so la",                     "locative"),
    ("nous sommes au marché",             "anw bɛ súgu la",                 "locative_plur"),
    ("il est au champ",                   "a bɛ foro la",                   "locative"),
    ("elle est à l'école",                "a bɛ lakɔli la",                 "locative"),
    ("il n'est pas à la maison",          "a tɛ so la",                     "locative_neg"),
    ("je ne suis pas au marché",          "n tɛ súgu la",                   "locative_neg"),

    # ════════════════════════════════════════════════════════════════════
    # III. ÊTRE — QUALIFICATION
    # ════════════════════════════════════════════════════════════════════
    ("je suis belle",                     "n ka ɲuman",                     "qualitative"),
    ("le cheval est rapide",              "sòn ka téliman",                  "qualitative"),
    ("La maison est loin",                "so ka jàn",                      "qualitative"),
    ("Les maisons sont loin",             "sow ka jàn",                     "qualitative_plur"),
    ("l'eau est chaude",                  "jí ka bɛ̀lɛ",                    "qualitative"),
    ("la route est longue",               "sìra ka télé",                   "qualitative"),
    ("tu n'es pas bien",                  "i mán ɲuman",                    "qualitative_neg"),
    ("la maison n'est pas grande",        "so mán bàrika",                  "qualitative_neg"),
    ("la route était longue",             "sìra tùn ka télé",               "qualitative_past"),
    ("il n'était pas bien",               "a tùn mán ɲuman",                "qualitative_past_neg"),

    # ════════════════════════════════════════════════════════════════════
    # IV. ÊTRE — ÉTAT STATIF
    # ════════════════════════════════════════════════════════════════════
    ("je suis sûre de ça",                "n lalen dòn à la",               "statif"),
    ("Les maisons sont sûres",            "sow lalen dòn",                  "statif_plur"),
    ("C'est cuit",                        "o mɔnna",                        "past_intransitive"),
    ("Les viandes sont cuites",           "sɔgɔw mɔnna",                   "past_intransitive_plur"),
    ("il est parti",                      "o fáɲira",                       "past_intransitive"),
    ("elle est tombée",                   "a bɛnna",                        "past_intransitive"),
    ("ils sont arrivés",                  "ùw sera",                        "past_intransitive_plur"),

    # ════════════════════════════════════════════════════════════════════
    # V. AVOIR — PASSÉ TRANSITIF
    # ════════════════════════════════════════════════════════════════════
    ("Musa a mangé le riz",               "Musa yé màlo dún",               "past_transitive"),
    ("J'ai acheté des pagnes",            "n yé fìníw sàn",                 "past_transitive"),
    ("tu as vu l'homme",                  "i yé cɛ yé",                     "past_transitive"),
    ("elle a bu de l'eau",                "a yé jí mìn",                    "past_transitive"),
    ("nous avons mangé le riz",           "anw yé màlo dún",                "past_transitive_plur"),
    ("ils ont construit la maison",       "ùw yé so lá",                    "past_transitive_plur"),
    ("L'homme que tu as vu hier est parti", "i yé cɛ mìn yé kúnùn, o fáɲira", "relative_topic"),

    # ════════════════════════════════════════════════════════════════════
    # VI. AVOIR — POSSESSION MATÉRIELLE
    # ════════════════════════════════════════════════════════════════════
    ("J'ai de l'argent",                  "wárí bɛ n bóló",                 "noun_phrase_have_material"),
    ("il a une voiture",                  "[voiture] bɛ a bóló",            "noun_phrase_have_material"),
    ("Ils ont des voitures",              "móbiliw bɛ ùw bóló",             "noun_phrase_have_material_plur"),
    ("tu as un couteau",                  "mùru bɛ i bóló",                 "noun_phrase_have_material"),
    ("Quel âge as-tu ?",                  "sán jóli bɛ i bóló  ?",          "noun_phrase_have_age"),
    ("elle n'a pas d'argent",             "wárí tɛ a bóló",                 "noun_phrase_have_material_neg"),

    # ════════════════════════════════════════════════════════════════════
    # VII. AVOIR — POSSESSION ABSTRAITE
    # ════════════════════════════════════════════════════════════════════
    ("J'ai un frère",                     "bálimakɛ bɛ n fɛ",               "noun_phrase_have_abstract"),
    ("J'ai une sœur",                     "bálimamuso bɛ n fɛ",             "noun_phrase_have_abstract"),
    ("J'ai des frères et sœurs",          "bálimaw bɛ n fɛ",                "noun_phrase_have_abstract_plur"),
    ("J'aime manger",                     "n bɛ dún fɛ",                    "volitif"),
    ("il a de la chance",                 "diyaɲɔgɔnya bɛ a fɛ",           "noun_phrase_have_abstract"),
    ("elle n'a pas de frère",             "bálimakɛ tɛ a fɛ",               "noun_phrase_have_abstract_neg"),

    # ════════════════════════════════════════════════════════════════════
    # VIII. EXISTENCE PURE
    # ════════════════════════════════════════════════════════════════════
    ("Il y'a du pain",                    "búuru bɛ",                       "existential_absolute"),
    ("il y a un problème",                "[problème] bɛ",                  "existential_absolute"),
    ("il y a du travail",                 "baaraw bɛ",                      "existential_absolute"),
    ("il n'y a pas de pain",              "búuru tɛ",                       "existential_absolute_neg"),
    ("il n'y a personne dans le village", "mɔgɔ tɛ dugu kɔnɔ",             "existential_localized_neg"),

    # ════════════════════════════════════════════════════════════════════
    # IX. EXISTENCE LOCALISÉE
    # ════════════════════════════════════════════════════════════════════
    ("Il y a de l'eau dans la bouteille", "jí bɛ bútèli kɔnɔ",             "existential_localized"),
    ("Il y a des gens ici",               "mɔgɔw bɛ yàn",                   "existential_localized"),
    ("Il n'y a personne ici",             "mɔgɔw tɛ yàn",                   "existential_localized_neg"),
    ("il y a des enfants dans la maison", "dɔgɔw bɛ so kɔnɔ",              "existential_localized"),
    ("il n'y a pas d'eau dans la bouteille", "jí tɛ bútèli kɔnɔ",          "existential_localized_neg"),

    # ════════════════════════════════════════════════════════════════════
    # X. PRÉSENTATIF / IDENTIFICATOIRE
    # ════════════════════════════════════════════════════════════════════
    ("C'est moi",                         "n dòn",                          "identificatory"),
    ("C'est moi Hawa",                    "n de dòn, Hawa",                 "identificatory_appos"),
    ("Ce n'est pas moi",                  "n tɛ",                           "identificatory_neg"),
    ("c'est Musa",                        "Musa dòn",                       "presentative"),
    ("Ce sont mes frères et sœurs",       "n bálimaw dòn",                  "presentative_plur"),
    ("c'est lui",                         "a dòn",                          "identificatory"),
    ("ce n'est pas lui",                  "a tɛ",                           "identificatory_neg"),
    ("c'est nous",                        "anw dòn",                        "identificatory"),

    # ════════════════════════════════════════════════════════════════════
    # XI. DÉICTIQUE
    # ════════════════════════════════════════════════════════════════════
    ("Voilà la voiture",                  "[voiture] félé",                  "deictique"),
    ("Voilà les voitures",                "[voiture]w félé",                 "deictique_plur"),
    ("Voilà Musa",                        "Musa félé",                       "deictique"),
    ("Voilà l'eau",                       "jí félé",                         "deictique"),

    # ════════════════════════════════════════════════════════════════════
    # XII. INTERROGATIVES OUI/NON
    # ════════════════════════════════════════════════════════════════════
    ("Parles-tu bambara ?",               "i bɛ bambara kúma wà ?",         "interrogative"),
    ("Allez-vous au marché ?",            "[vous] bɛ wá súgu la wà ?",      "interrogative_motion"),
    ("Est-ce que ça va vraiment bien ?",  "o bɛ wá kóɲuman nɛ̀fɛ wà ?",    "interrogative_question_marker"),
    ("Est-ce que tes frères et sœurs sont ici ?", "Yala i bálimakɛw ni i balimamusow bɛ yàn wà ?", "interrogative_question_marker"),
    ("Tu veux du poisson ou bien tu veux de la viande ?", "i bɛ jɛ́gɛ ŋàniya wàwá i bɛ sògo ŋàniya ?", "interrogative_alternative"),
    ("Tu veux des arachides ou bien tu veux des patates douces ?", "i bɛ tìgan ŋàniya wàwá i bɛ [patate douce] ŋàniya ?", "interrogative_alternative"),
    ("il mange ?",                        "a bɛ dún wà ?",                  "interrogative"),
    ("tu as de l'argent ?",               "wárí bɛ i bóló wà ?",            "interrogative"),
    ("elle est à la maison ?",            "a bɛ so la wà ?",                "interrogative"),

    # ════════════════════════════════════════════════════════════════════
    # XIII. QUESTIONS DE CONTENU
    # ════════════════════════════════════════════════════════════════════
    ("Qui fait la bagarre ?",             "jɔn bɛ bìlen dími ?",            "content_question_who"),
    ("Qui aimez-vous ?",                  "[vous] bɛ jɔn kànu ?",           "content_question_who_inversion"),
    ("Qui aiment-ils ?",                  "[ils] bɛ jɔn kànu ?",            "content_question_who_inversion"),
    ("Tu manges quoi ?",                  "i bɛ mún dún ?",                 "content_question_what"),
    ("Tu veux quoi ?",                    "i bɛ mún ŋàniya ?",              "content_question_what"),
    ("il mange quoi ?",                   "a bɛ mún dún ?",                 "content_question_what"),
    ("Où allez-vous ?",                   "[vous] bɛ wá mín ?",             "content_question_where"),
    ("où est Musa ?",                     "Musa bɛ mín ?",                  "content_question_where"),
    ("Quand viens-tu ?",                  "i bɛ nà túma jùmɛn ?",           "content_question_when"),
    ("quand pars-tu ?",                   "i bɛ táa túma jùmɛn ?",          "content_question_when"),
    ("Pourquoi pleures-tu ?",             "i bɛ kàsi mún kósɔn ?",          "content_question_why"),
    ("pourquoi il mange ?",               "a bɛ dún mún kósɔn ?",           "content_question_why"),
    ("Comment t'appelles-tu ?",           "i bɛ kánbìla cógo dǐ ?",         "content_question_how"),
    ("Comment cela se fait-il ?",         "o bɛ dími cógo dǐ ?",            "content_question_how"),
    ("Quelle femme ?",                    "númanfɛlaka jùmɛn ?",            "content_question_which_noun"),
    ("Tu veux quelle maison ?",           "i bɛ só jùmɛn ŋàniya ?",         "content_question_which"),
    ("Ça coûte combien ?",                "wárí jóli bɛ à la ?",            "content_question_how_much"),
    ("tu as combien d'enfants ?",         "dɔgɔ jóli bɛ i fɛ ?",           "content_question_how_much"),

    # ════════════════════════════════════════════════════════════════════
    # XIV. SIMPLE PRÉSENT
    # ════════════════════════════════════════════════════════════════════
    ("je mange du riz",                   "n bɛ màlo dún",                  "simple"),
    ("tu bois de l'eau",                  "i bɛ jí mìn",                    "simple"),
    ("il parle bambara",                  "a bɛ bambara kúma",              "simple"),
    ("elle chante",                       "a bɛ donkili dón",               "simple"),
    ("nous travaillons",                  "anw bɛ báara la",                "simple_plur"),
    ("ils mangent du riz",                "ùw bɛ màlo dún",                 "simple_plur"),
    ("je ne mange pas",                   "n tɛ dún",                       "simple_neg"),
    ("il ne parle pas bambara",           "a tɛ bambara kúma",              "simple_neg"),

    # Volitif
    ("je veux manger",                    "n bɛ dún fɛ",                    "volitif"),
    ("il veut partir",                    "a bɛ táa fɛ",                    "volitif"),
    ("je ne veux pas manger",             "n tɛ dún fɛ",                    "volitif_neg"),

    # Progressif
    ("je suis en train de manger",        "n bɛ kà dún",                    "progressif"),
    ("il est en train de parler",         "a bɛ kà kúma",                   "progressif"),

    # Futur
    ("je vais manger",                    "n bɛ na dún",                    "futur"),
    ("il va partir",                      "a bɛ na táa",                    "futur"),
    ("je n'irai pas",                     "n tɛ na táa",                    "futur_neg"),

    # Passé négatif
    ("je n'ai pas mangé",                 "n ma màlo dún",                  "past_neg"),
    ("il n'a pas parlé",                  "a ma kúma",                      "past_neg"),

    # Habitude
    ("il mangeait du riz",                "a tùn bɛ màlo dún",              "habitude"),
    ("je travaillais",                    "n tùn bɛ báara la",              "habitude"),

    # ════════════════════════════════════════════════════════════════════
    # XV. VERBE SÉRIEL / COMPLEXE
    # ════════════════════════════════════════════════════════════════════
    ("Je te donne de l'argent pour cuisiner et manger",
     "n bɛ wárí di i ma walasa ka [cuisiner] ani ka dún",
     "simple_purposive_coordinated"),

    ("Je veux manger avec mon mari et ma fille qui est malade",
     "n bɛ ŋàniya ka dún ni n fúrucɛ ni n mùsona jànkarotɔ",
     "complex_comitative_relative"),

    ("il donne le livre à l'enfant",      "a bɛ kitabu di dɔgɔ ma",        "verb_serial_dative"),

    # ════════════════════════════════════════════════════════════════════
    # XVI. PRIVATIF
    # ════════════════════════════════════════════════════════════════════
    ("sans moi",                          "n kɔ",                           "privative_pron"),
    ("sans moi, tu ne pourras pas partir","n kɔ, i tɛ na se ka táa",        "privative_clause"),
    ("Ce légume est sans cuisson",        "[légume] [cuisson]tan dòn",       "privative_noun"),
    ("il est parti sans avertir",         "a fáɲira [avertir]bali",         "privative_verb"),
    ("sans eau",                          "jítan",                           "privative_noun"),
    ("sans argent",                       "wárítan",                         "privative_noun"),

    # ════════════════════════════════════════════════════════════════════
    # XVII. RELATIF / TOPIC
    # ════════════════════════════════════════════════════════════════════

    ("Une femme qui a eu un enfant ne peut pas abandonner son enfant",
     "númanfɛlaka mìn yé dén sɔrɔ, o tɛ se ka a dén láfìli",
     "relative_topic_neg"),

    ("Les djihadistes montrent clairement qu'ils sont les maîtres du jeu",
     "djihadistew bɛ jàbi ko ùw yé [jeu] ka kuntigɛw yé",
     "equative_relative"),

    # ════════════════════════════════════════════════════════════════════
    # XVIII. COMITATIVE / RÉCIPROQUE
    # ════════════════════════════════════════════════════════════════════
    ("je suis avec mon mari",             "n ni n cɛ dòn",                  "comitative"),
    ("Je suis avec la fille du frère de mon ami",
     "n ni n terikɛ balimakɛ denmuso dòn",
     "comitative_genitive"),
    ("je mange avec mon ami",             "n bɛ dún ni n terikɛ",           "comitative_action"),
    ("nous nous aimons",                  "an bɛ ɲɔgɔn kànu",              "reciprocal"),
    ("ils se battent",                    "ùw bɛ ɲɔgɔn fàli",              "reciprocal"),
    ("nous nous parlons",                 "anw bɛ ɲɔgɔn kúma",             "reciprocal"),

    # ════════════════════════════════════════════════════════════════════
    # XIX. IMPÉRATIF / PROHIBITIF
    # ════════════════════════════════════════════════════════════════════
    ("mange !",                           "dún",                             "imperative"),
    ("pars !",                            "táa",                             "imperative"),
    ("viens !",                           "nà",                              "imperative"),
    ("parle bambara !",                   "bambara kúma",                    "imperative"),
    ("ne mange pas !",                    "kàna dún",                        "prohibitive"),
    ("ne pars pas !",                     "kàna táa",                        "prohibitive"),
    ("ne parle pas !",                    "kàna kúma",                       "prohibitive"),

    # ════════════════════════════════════════════════════════════════════
    # XX. APPARTENANCE / SYNTAGME NOMINAL
    # ════════════════════════════════════════════════════════════════════
    ("ce sac est pour ma fille",          "nin jɔ̀bɔrɔ in yé n mùsoma de ta ye", "ownership"),
    ("la maison de mon ami",              "n terikɛ ka so",                  "noun_phrase_alienable"),
    ("la participation citoyenne et démocratique",
     "angageman sosiyali ani lademokarasi",
     "noun_phrase_coord"),
    ("La décision finale de la constitution du Mali",
     "Mali dácogo làtigɛ [final]",
     "noun_phrase_genitive"),
    ("le champ du vieux",                 "kɔrɔba ka foro",                  "noun_phrase_genitive"),
    ("la mère de l'enfant",               "dɔgɔ ba",                         "noun_phrase_genitive"),
    ("la maison de Musa",                 "Musa ka so",                       "noun_phrase_genitive"),

    # ════════════════════════════════════════════════════════════════════
    # XXI. INFINITIF
    # ════════════════════════════════════════════════════════════════════
    ("Promouvoir le bambara et les autres langues nationales",
     "ka bambara wɛ́rɛ nɛ̀n Nasiyonali kìsɛya",
     "infinitive"),
    ("manger du riz",                     "ka màlo dún",                     "infinitive"),
    ("parler bambara",                    "ka bambara kúma",                  "infinitive"),
    ("ne pas manger",                     "kàna dún",                        "infinitive_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXII. PHRASES LONGUES / COMPLEXES
    # ════════════════════════════════════════════════════════════════════
    ("Promouvoir le bambara et les autres langues nationales, inclusion sociale, meilleure circulation de l'information",
     "ka bambara wɛ́rɛ nɛ̀n Nasiyonali kìsɛya, [inclusion] sosiyali, kùnnafoni fìsamannci ka sírakankasaara",
     "infinitive_list"),

    ("Mais la situation reste très incertaine au Mali, qui s'enfonce toujours plus dans le chaos",
     "nka dábolo bɛ [incertain] lámàra Mali kɔnɔ, min bɛ ɲàgàmi fɔlɔ kɔfɛ cànla la",
     "complex_relative"),

    ("Il existe un frein politique lié aux logiques de pouvoir des élites qui veulent maintenir la masse populaire à l'écart de la gouvernance",
     "[frein] [politique] bɛ, min bɛ [élite]w ka [logique]w la, minw bɛ ŋàniya ka [masse populaire] bìla [gouvernance] kɔrɔ",
     "existential_nominal_complex"),

    ("Chercheur associé à l'Institut français des relations internationales, Thierry Vircoulon revient pour 20 Minutes sur la situation explosive au Mali",
     "Thierry Vircoulon, IFRIw la kùndafɔrɔ, bɛ segin 20 Minutes ma Mali dábolo [explosif] kan",
     "complex_noun_phrase"),

    ("L'université de Kumamoto recrute actuellement des participants pour le Programme d'apprentissage coopératif japonais, ainsi que des étudiants japonais pour la soutenir",
     "Kumamoto lakɔli bɛ kàlandenw wele sísàn japɔnikan [coopératif] kàlanso la, ani japɔni kalandenw ka a dɛmɛ",
     "complex_purposive"),

    # ════════════════════════════════════════════════════════════════════
    # XXIII. CORRECTIFS SESSION 2026-06
    # ════════════════════════════════════════════════════════════════════

    # Locatif ADV pur avec être (ici/là ROOT, pas existentiel)
    ("les gens sont ici",                "ɲàmakalaw bɛ yàn",               "locative_adv"),
    ("il est là",                        "a bɛ yèn",                       "locative_adv"),

    # Numéraux cardinaux — sujet et objet
    ("deux hommes sont partis",          "mɔ̀gɔw fila fáɲira",             "nummod_subject"),
    ("trois enfants mangent du riz",     "dénw saba bɛ iri dún",            "nummod_subject"),
    ("cinq femmes sont ici",             "númanfɛlakaw duuru bɛ yàn",       "nummod_locative"),
    ("j'ai acheté deux livres",          "n yé kìtabuw fila sàn",           "nummod_object"),

    # Verbe intransitif ACTION en INTRANS_SC (báara = travailler)
    ("je suis en train de travailler",   "n bɛ kà báara kɛ",               "intrans_action_progressif"),
    ("je ne travaille pas",              "n tɛ báara la",                   "intrans_action_present_neg"),
    ("j'ai travaillé",                   "n yé báara kɛ",                   "intrans_action_passe_pos"),
    ("je n'ai pas travaillé",            "n ma báara kɛ",                   "intrans_action_passe_neg"),
    ("je ne travaillais pas",            "n tùn tɛ báara la",               "intrans_action_hab_neg"),
    ("je travaillerai",                  "n bɛ na báara kɛ",                "intrans_action_futur"),

    # Subordonnant temporel (quand → tuma min antéposé)
    ("quand il vient",                   "tuma min a bɛ nà",                "temporal_subordinator"),

    # Discours rapporté + comparatif (plus ADJ que X → ka ADJ ka tɛmɛ X kan)
    ("les bambara disent que la raison de la venue de quelqu'un est plus importante que soi-même",
     "bambaraw bɛ fɔ ko mɔ̀gɔ dɔ jɔ̀kun ka kólogirinman ka tɛmɛ yɛrɛ kan",
     "reported_comparative"),
]


# ── RUNNER ────────────────────────────────────────────────────────────────────
def run_tests(translate_fn=None):
    import csv, datetime
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

    for i, (phrase, expected, category) in enumerate(TEST_CASES, 1):
        print(f"[{i:02d}/{total}] {category}")
        print(f"  FR  : {phrase}")

        if translate_fn:
            try:
                result = translate_fn(phrase).strip()
                ok = result == expected.strip()
                if ok:
                    passed += 1
                    print(f"  ✅  : {result}")
                    results.append((i, category, phrase, expected, result, 'PASS'))
                else:
                    failed.append((i, category, phrase, expected, result))
                    print(f"  ❌  : {result}")
                    results.append((i, category, phrase, expected, result, 'FAIL'))
            except Exception as e:
                err = f"ERREUR: {e}"
                failed.append((i, category, phrase, expected, err))
                print(f"  💥  : {e}")
                results.append((i, category, phrase, expected, err, 'ERROR'))
        else:
            results.append((i, category, phrase, expected, '', ''))
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
                print(f"  ❌ {cat}: {fail_cats[cat]}/{cats.get(cat,0)}")
        print(f"{'='*72}\n")
    else:
        print(f"{'='*72}")
        print(f"  {total} phrases — {len(cats)} catégories\n")
        print("  Catégories :")
        for cat in sorted(cats):
            print(f"    {cats[cat]:2d}  {cat}")
        print(f"\n  Pour tester : python test_phrases.py --run")
        print(f"{'='*72}\n")

    # ── Export CSV ─────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"resultats_bambara_{ts}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['#', 'categorie', 'phrase_fr', 'bambara_attendu', 'bambara_obtenu', 'statut'])
        for row in results:
            writer.writerow(row)
    print(f"  CSV exporté : {csv_path}\n")
    return csv_path


if __name__ == '__main__':
    import sys
    if '--run' in sys.argv:
        try:
            sys.path.insert(0, '.')
            from pipeline.translation_engine import TranslationEngine
            from kg.neo4j_client import Neo4jClient
            db     = Neo4jClient()
            engine = TranslationEngine(db)
            run_tests(translate_fn=lambda s: engine.translate(s)['bambara'])
        except ImportError as e:
            print(f"Moteur non trouvé ({e})\n")
            run_tests()
        except Exception as e:
            print(f"Erreur d'initialisation : {e}\n")
            run_tests()
    else:
        run_tests()