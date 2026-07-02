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
    ("J'aime manger",                     "n bɛ dúnli kànuya",              "volitif"),
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
    ("C'est moi Hawa",                    "n de Hawa yé",                   "identificatory_appos"),
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
    ("peut-il manger de la viande ?",     "a bɛ se ka sògo dún wà ?",       "interrogative_modal_xcomp"),

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
    ("je veux manger",                    "n bɛ ŋàniya ka dúnli kɛ",        "volitif"),
    ("il veut partir",                    "a bɛ ŋàniya ka táa",             "volitif"),
    ("je ne veux pas manger",             "n tɛ ŋàniya ka dúnli kɛ",        "volitif_neg"),

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
    ("Ce sac est pour moi",              "nin jɔ̀bɔrɔ in yé n ta ye",          "ownership_pron"),
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

    # ════════════════════════════════════════════════════════════════════
    # XXIV. CORRECTIFS SESSION 2026-06-12
    # ════════════════════════════════════════════════════════════════════

    # Optatif / subjonctif : que + Mood=Sub → S ka (O) V
    ("que tu viennes",                     "i ka nà",                         "optatif"),
    ("que Moussa mange",                   "Moussa ka dún",                   "optatif"),
    ("que Dieu t'aide",                    "Ala ka i dɛmɛ",                   "optatif_coi"),

    # Venir de + lieu (bɔra) — PROPN sans 'la', NOUN avec 'la'
    ("Je viens de Bamako",                 "n bɔra Bamako",                   "venir_de_propn"),
    ("Je viens de l'école",                "n bɔra kàlankɛyɔrɔ la",          "venir_de_noun"),

    # Venir de + verbe (passé récent) → bɔra ka + verbe avec transitivité
    ("Je viens de manger",                 "n bɔra ka dumuni kɛ",             "venir_de_verbe"),
    ("Il vient de partir",                 "a bɔra ka táa",                   "venir_de_verbe"),

    # Verbe coordonné avec oblique locatif (oblique avant la clause coordonnée)
    ("elle alla au village et demande des infos",
     "a yé wá dùgu la wa a yé ɛnfo ɲɛ́juguya",
     "conj_avec_locatif"),

    # Verbe sériel + verbe coordonné sur le xcomp
    ("elle alla trouver et demande des informations",
     "a táara ɲɛ́sɔ̀rɔ wa a yé kùnnafoniw ɲɛ́juguya",
     "verb_serial_conj"),

    # Comitatif + verbe de mouvement ABSOLU (pas de 'li kɛ', pas de doublon ni/yé)
    ("il est venu avec moi",               "a nàra ni n yé",                  "comitative_motion"),
    ("elle travaille avec lui",            "a bɛ báara ni a yé",              "comitative_travail"),

    # Comitatif + adjectif sur le nom comitatif
    ("Il est venu avec une dentition complète",
     "a nàra ni dákolon dafaleninman yé",
     "comitative_adj"),

    # ════════════════════════════════════════════════════════════════════
    # XXV. SYNTAGME NOMINAL TEMPOREL — CORRECTIFS 2026-06-13
    # ════════════════════════════════════════════════════════════════════

    # Préposition temporelle + quantifier + nom → kabini préfixé, nom pluriel, dɔw postposé
    ("Depuis quelques mois",              "kabini sélidenninkalow dɔw",        "temporal_np_depuis"),
    ("depuis quelques jours",             "kabini tilew dɔw",                  "temporal_np_depuis"),
    ("depuis quelques années",            "kabini sanw dɔw",                   "temporal_np_depuis"),

    # ════════════════════════════════════════════════════════════════════
    # XXVI. SUBORDONNÉE COMPLÉTIVE — ko + clause (ccomp)
    # ════════════════════════════════════════════════════════════════════

    # Verbe cognitif savoir/dɔn — ko + clause équative
    ("sais-tu que je suis un enfant ?",   "i bɛ dɔ́n ko n yé dén yé wà ?",    "ccomp_interrogative"),
    ("je sais que tu es mon ami",         "n bɛ dɔ́n ko i yé n terikɛ yé",     "ccomp_savoir"),
    ("ils savent que nous sommes ici",    "ùw bɛ dɔ́n ko anw bɛ yàn",          "ccomp_savoir_locatif"),

    # Verbe de parole dire — présent : S ko [ccomp] (sans TAM ni verbe dire)
    ("il dit qu'il mange",                "a ko a bɛ dún",                      "ccomp_dire"),
    ("ils disent que la route est longue","ùw ko sìra ka télé",                  "ccomp_dire_qualitative"),
    # Passé : S TAM VERBE ko [ccomp] (chemin normal conservé)
    ("elle a dit que Musa est parti",     "a yé fɔ ko Musa fáɲira",             "ccomp_dire_passe"),

    # ════════════════════════════════════════════════════════════════════
    # XXVII. MODAL — VARIATIONS SUR "POUVOIR / SE KA"
    # ════════════════════════════════════════════════════════════════════

    ("nous pouvons partir",               "anw bɛ se ka táa",                  "modal_pouvoir"),
    ("il ne peut pas manger",             "a tɛ se ka dún",                    "modal_pouvoir_neg"),
    ("elle peut venir",                   "a bɛ se ka nà",                     "modal_pouvoir"),
    ("tu ne peux pas partir",             "i tɛ se ka táa",                    "modal_pouvoir_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXVIII. PASSÉ INTRANSITIF NÉGATIF
    # ════════════════════════════════════════════════════════════════════

    ("ils ne sont pas arrivés",           "ùw ma se",                          "past_intransitive_neg"),
    ("il n'est pas tombé",                "a ma bɛn",                          "past_intransitive_neg"),
    ("elle n'est pas venue",              "a ma nà",                           "past_intransitive_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXIX. HABITUDE NÉGATIVE
    # ════════════════════════════════════════════════════════════════════

    ("elle ne mangeait pas",              "a tùn tɛ dún",                      "habitude_neg"),
    ("nous ne parlions pas bambara",      "anw tùn tɛ bambara kúma",           "habitude_neg_plur"),
    ("il n'avait pas mangé",              "a tùn ma dún",                      "passe_anterieur_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXX. RÈGLES SPÉCIALISÉES (Rules 1-9)
    # ════════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════════
    # RÈGLES SPÉCIALISÉES (Rules 1-9)
    # ════════════════════════════════════════════════════════════════════

    # Rule 1: Conditional + question → ends with 'dun ?' not 'wà ?'
    ("Si j'ai du courage, saurait-on ?",                    "n bɛ gara dun, a mán se dun ?",         "rule1_conditional_question"),

    # Rule 2: Prohibitive + object → 'kàna [object] V'
    ("Ne mange pas le riz",                                 "kàna iri dún",                         "rule2_prohibitive_object"),

    # Rule 3: Temporal + passé simple avoir + possessive
    ("quand il eut ton appel",                              "tuma min a yé i ka wéle ɲóro",         "rule3_temporal_avoir_possessive"),

    # Rule 4: Temporal + passive passé simple
    ("Quand cela fut fait",                                 "tuma min o tùn yògorolen dòn",         "rule4_temporal_passive"),

    # Rule 5: Fixed phrase 'Ainsi donc'
    ("Ainsi donc",                                          "ola sa",                               "rule5_fixed_phrase"),

    # Rule 6: ne...que restrictive → 'S TAM foyi yé ni ATTR tɛ'
    ("tu ne serais qu'un pleutre",                          "i tɛ foyi yé ni sègɛ tɛ",             "rule6_restrictive"),

    # Rule 7: Qu'est-ce que = mún S TAM V ka O V_ACT [obliques] (expletive: no S)
    ("Qu'est-ce qu'il pourrait t'arriver là-bas ?",         "mún bɛ ka sé i ma yèn jùkɔ́rɔw",      "rule7_quest_ce_que"),
    ("Qu'est-ce qu'il peut faire ?",                         "A bɛ se ka mun kɛ ?",                  "rule7_quest_ce_que_real_subject"),

    # Rule 8: Complex relative with reflexive + correct clause splitting
    ("Toi qui prends l'ennemi vivant",                      "i bɛ júgu ɲɛ́nama",                    "rule8_relative_splitting"),

    # Rule 9: valoir la peine idiom (follows general SOV rule)
    ("cela vaut la peine de prendre un fusil",              "o bɛ buntu cɛ wà ka sàn",              "rule9_valoir_peine_sov"),

    # Additional variants for better coverage
    ("tu ne serais que jaloux",                             "i tɛ foyi yé ni jiliya tɛ",           "rule6_restrictive_adj"),

    # ════════════════════════════════════════════════════════════════════
    # XXXI. RÉFLEXIF ABSOLU — arbre de décision (sessions 2026-06-15)
    # ════════════════════════════════════════════════════════════════════
    # Structure : S TAM refl_pron [yɛrɛ] V
    # Cat. Actif (volontaire, yɛrɛ=False par défaut)
    ("il s'est lavé",                "a yé a jó",                          "refl_actif_passe"),
    ("elle se lave",                 "a bɛ a jó",                          "refl_actif_present"),

    # Cat. Actif + emphase explicite 'lui même' → yɛrɛ=True
    ("il s'est lavé lui même",       "a yé a yɛrɛ jó",                    "refl_actif_emphase"),

    # Cat. Accidentel (involontaire, yɛrɛ=True)
    ("il s'est blessé",              "a yé a yɛrɛ màjógin",               "refl_accidentel_passe"),

    # Posture (is_refl_Subjective, Cat. Actif)
    ("il s'est assis",               "a yé a sìgi",                        "refl_posture_passe"),

    # Idiomatique + xcomp locatif → S yé refl_pron yɛrɛ V_MAIN V_ACT la
    ("il s'est mis à pleurer",       "a yé a yɛrɛ bìla kàsi la",          "refl_idiom_locatif"),

    # ════════════════════════════════════════════════════════════════════
    # XXXII. CORRECTIONS SESSION 2026-06-17
    # ════════════════════════════════════════════════════════════════════

    # Pronom objet indirect (iobj) à la 1re personne
    ("Il me parle",                  "a bɛ kúma n yé",                    "simple_iobj_present"),
    ("elle me donne un livre",       "a bɛ kìtabu di n ma",               "verb_serial_dative_me"),

    # Question de contenu + modal + xcomp (pouvoir faire ?)
    ("Qu'est-ce qu'il pourrait faire ?",  "a bɛ se ka mún kɛ ?",          "content_question_modal_xcomp"),

    # ════════════════════════════════════════════════════════════════════
    # XXXIII. AVOIR AU FUTUR — POSSESSION FUTURE
    # ════════════════════════════════════════════════════════════════════

    ("Tu auras une voiture",         "móbili bɛ na i bóló",               "future_have_material"),
    ("Elle aura de l'argent",        "wárí bɛ na a bóló",                 "future_have_material"),
    ("J'aurai un frère",             "bálimakɛ bɛ na n fɛ",               "future_have_abstract"),
    ("Il aura du succès",            "[succès] bɛ na a fɛ",               "future_have_abstract"),
    ("Nous aurons des enfants",      "dénw bɛ na anw fɛ",                 "future_have_abstract_plur"),
    ("ils n'auront pas de voiture",  "móbili tɛ na ùw bóló",              "future_have_material_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXXIV. AVOIR EU — ACQUISITION PASSÉE (yé ... sɔrɔ)
    # ════════════════════════════════════════════════════════════════════

    ("j'ai eu un mari",              "n yé cɛ sɔrɔ",                      "past_transitive_avoir_eu"),
    ("elle a eu des enfants",        "a yé dénw sɔrɔ",                    "past_transitive_avoir_eu_plur"),
    ("tu as eu de la chance",        "i yé diyaɲɔgɔnya sɔrɔ",            "past_transitive_avoir_eu"),
    ("il a eu un problème",          "a yé [problème] sɔrɔ",              "past_transitive_avoir_eu"),
    ("nous avons eu des difficultés","anw yé [difficulté]w sɔrɔ",         "past_transitive_avoir_eu_plur"),

    # ════════════════════════════════════════════════════════════════════
    # XXXV. POSSESSION PASSÉE — HABITUDE (tùn bɛ ... bóló / fɛ)
    # ════════════════════════════════════════════════════════════════════

    ("j'avais une voiture",          "móbili tùn bɛ n bóló",              "past_have_material"),
    ("il avait de l'argent",         "wárí tùn bɛ a bóló",                "past_have_material"),
    ("elle n'avait pas de frère",    "bálimakɛ tùn tɛ a fɛ",              "past_have_abstract_neg"),
    ("nous avions un champ",         "foro tùn bɛ anw bóló",              "past_have_material_plur"),
    ("tu avais de la chance",        "diyaɲɔgɔnya tùn bɛ i fɛ",          "past_have_abstract"),

    # ════════════════════════════════════════════════════════════════════
    # XXXVI. OPTATIF ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("Que Dieu vous bénisse",        "Ala ka aw dɛmɛ",                    "optatif_plur"),
    ("Qu'il vienne",                 "a ka nà",                           "optatif"),
    ("Qu'ils partent",               "ùw ka táa",                         "optatif_plur"),
    ("Que la paix règne",            "lafiɲa ka kɛ",                      "optatif_abs"),
    ("Que tu réussisses",            "i ka dɔngɛ",                        "optatif"),
    ("Que nous mangions ensemble",   "anw ka dún ɲɔgɔn fɛ",              "optatif_plur"),

    # ════════════════════════════════════════════════════════════════════
    # XXXVII. COORDINATION VERBALE ÉTENDUE
    # ════════════════════════════════════════════════════════════════════

    # S TAM V1 wa S TAM V2 (présent)
    ("il mange et boit",             "a bɛ dún wa a bɛ mìn",              "conj_verb_present"),
    ("elle chante et danse",         "a bɛ donkili dón wa a bɛ [danser]", "conj_verb_present"),
    # Passé coordonné
    ("il est venu et a mangé",       "a nàra wa a yé dún",                "conj_verb_passe"),

    # ════════════════════════════════════════════════════════════════════
    # XXXVIII. MODAL ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("tu peux manger",               "i bɛ se ka dún",                    "modal_pouvoir"),
    ("nous ne pouvons pas venir",    "anw tɛ se ka nà",                   "modal_pouvoir_neg"),
    ("elles peuvent travailler",     "ùw bɛ se ka báara kɛ",              "modal_pouvoir_plur"),
    ("il pouvait partir",            "a tùn bɛ se ka táa",                "modal_pouvoir_past"),
    ("peut-elle partir ?",           "a bɛ se ka táa wà ?",               "modal_pouvoir_interrogative"),
    ("il ne pouvait pas manger",     "a tùn tɛ se ka dún",                "modal_pouvoir_past_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XXXIX. IOBJ / DATIF ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("il me donne de l'argent",      "a bɛ wárí di n ma",                 "verb_serial_dative"),
    ("elle lui parle",               "a bɛ kúma a yé",                    "simple_iobj"),
    ("nous leur donnons du riz",     "anw bɛ màlo di ùw ma",              "verb_serial_dative_plur"),
    ("je te donne un livre",         "n bɛ kìtabu di i ma",               "verb_serial_dative"),
    ("il nous parle",                "a bɛ kúma anw yé",                  "simple_iobj_plur"),

    # ════════════════════════════════════════════════════════════════════
    # XL. RÉFLEXIF ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("il se lève",                   "a bɛ a tín",                        "refl_posture_present"),
    ("il s'est levé",                "a yé a tín",                        "refl_posture_passe"),
    ("elle s'est trompée",           "a yé a yɛrɛ tɔn",                  "refl_accidentel_passe"),
    ("il s'est mis à travailler",    "a yé a yɛrɛ bìla báara la",        "refl_idiom_locatif"),
    ("elle se regarde",              "a bɛ a yɛrɛ lajé",                  "refl_actif_present"),
    ("nous nous préparons",          "anw bɛ anw yɛrɛ labɛn",            "refl_actif_present_plur"),
    ("elle se réveille",             "a bɛ a wuli",                       "refl_posture_present"),

    # ════════════════════════════════════════════════════════════════════
    # XLI. IMPÉRATIF ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("viens ici !",                  "nà yàn",                            "imperative_locative"),
    ("ne fais pas ça !",             "kàna o kɛ",                         "prohibitive_dem"),
    ("aide-moi !",                   "n dɛmɛ",                            "imperative_iobj"),
    ("ne pleure pas !",              "kàna kàsi",                         "prohibitive"),
    ("mange vite !",                 "dún joona",                          "imperative_adv"),
    ("ne bois pas l'eau !",          "kàna jí mìn",                       "prohibitive_object"),
    ("ne mange pas la viande !",     "kàna sògo dún",                     "prohibitive_object"),

    # ════════════════════════════════════════════════════════════════════
    # XLII. INTERROGATIVE ÉTENDUE
    # ════════════════════════════════════════════════════════════════════

    ("Est-ce qu'il mange ?",         "Yala a bɛ dún wà ?",                "interrogative_est_ce_que"),
    ("Est-ce qu'elle travaille ?",   "Yala a bɛ báara la wà ?",           "interrogative_est_ce_que"),
    ("a-t-il mangé ?",               "a yé dún wà ?",                     "interrogative_passe"),
    ("Travailles-tu ?",              "i bɛ báara la wà ?",                "interrogative"),
    ("est-il venu ?",                "a nàra wà ?",                       "interrogative_past_intrans"),
    ("Est-ce qu'ils sont arrivés ?", "Yala ùw sera wà ?",                 "interrogative_est_ce_que_passe"),

    # ════════════════════════════════════════════════════════════════════
    # XLIII. QUESTIONS DE CONTENU ÉTENDUES
    # ════════════════════════════════════════════════════════════════════

    ("Qui est-il ?",                 "a yé jɔn yé ?",                     "content_question_who_equative"),
    ("Qu'est-ce que tu fais ?",      "i bɛ mún kɛ ?",                     "content_question_what"),
    ("Quand est-il parti ?",         "a fáɲira túma jùmɛn ?",             "content_question_when_passe"),
    ("Pourquoi ne viens-tu pas ?",   "i tɛ nà mún kósɔn ?",               "content_question_why_neg"),
    ("Combien coûte ce livre ?",     "wárí jóli bɛ nin kìtabu la ?",      "content_question_how_much"),
    ("Qui a mangé ?",                "jɔn yé dún ?",                      "content_question_who_passe"),

    # ════════════════════════════════════════════════════════════════════
    # XLIV. RELATIVES ÉTENDUES
    # ════════════════════════════════════════════════════════════════════

    ("La femme qui mange",           "númanfɛlaka mìn bɛ dún",            "relative_subject"),
    ("Le livre que j'ai acheté",     "kìtabu mìn n yé sàn",               "relative_object"),
    ("L'homme qui est parti",        "mɔ̀gɔ mìn fáɲira",                  "relative_subject_past"),
    ("L'enfant qui pleure",          "dén mìn bɛ kàsi",                   "relative_subject_present"),
    ("La maison que nous avons construite", "so mìn anw yé lá",           "relative_object_past"),

    # ════════════════════════════════════════════════════════════════════
    # XLV. CCOMP ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("tu sais qu'il est parti ?",    "i bɛ dɔ́n ko a fáɲira wà ?",         "ccomp_interrogative_passe"),
    # dire au présent → S ko [ccomp] (TAM + verbe dire supprimés)
    ("elle dit qu'elle partira",     "a ko a bɛ na táa",                   "ccomp_dire_futur"),
    ("nous savons qu'il est là",     "anw bɛ dɔ́n ko a bɛ yèn",            "ccomp_locatif"),
    ("il dit qu'il ne mange pas",    "a ko a tɛ dún",                      "ccomp_dire_neg"),
    ("je sais que tu es là",         "n bɛ dɔ́n ko i bɛ yèn",              "ccomp_locatif"),
    ("ils disent qu'il est venu",    "ùw ko a nàra",                       "ccomp_dire_passe_ccomp"),
    # dire au passé → S TAM VERBE ko [ccomp] (chemin normal conservé)
    ("il a dit qu'il mangeait",      "a yé fɔ ko a bɛ dún",               "ccomp_dire_passe_root"),
    ("ils ont dit qu'il était parti","ùw yé fɔ ko a fáɲira",              "ccomp_dire_passe_root_plur"),

    # ════════════════════════════════════════════════════════════════════
    # XLVI. VENIR DE ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("Ils viennent du marché",       "u bɛ bɔ súgu la",                   "venir_de_noun_plur"),
    ("Elle vient de la mosquée",     "a bɛ bɔ sìlamɛdiinɛ la",           "venir_de_noun"),
    ("Nous venons de Bamako",        "anw bɛ bɔ Bamako",                  "venir_de_propn_plur"),
    ("Elle vient de travailler",     "a bɔra ka báara kɛ",                "venir_de_verbe"),
    ("tu viens d'où ?",              "i bɛ bɔ mín ?",                     "venir_de_interrogative"),

    # ════════════════════════════════════════════════════════════════════
    # XLVII. COMITATIVE ÉTENDU
    # ════════════════════════════════════════════════════════════════════

    ("il est venu avec ses enfants", "a nàra ni a dénw yé",               "comitative_plur_noun"),
    ("je mange avec ma famille",     "n bɛ dún ni n dakɔrɔbaw yé",        "comitative_action"),
    ("il est parti avec ses amis",   "a fáɲira ni a terilaw yé",          "comitative_passe"),
    ("elle travaille avec son mari", "a bɛ báara ni a cɛ yé",             "comitative_travail"),

    # ════════════════════════════════════════════════════════════════════
    # XLVIII. RÈGLES SPÉCIALISÉES — COUVERTURE ÉTENDUE
    # ════════════════════════════════════════════════════════════════════

    # Rule 2 : prohibitif + objet (variantes)
    ("Ne bois pas le lait",          "kàna nɔnɔ mìn",                     "rule2_prohibitive_object"),
    ("Ne prends pas ça",             "kàna o ta",                          "rule2_prohibitive_dem"),

    # Rule 6 : ne...que (variantes)
    ("tu n'es qu'un enfant",         "i tɛ foyi yé ni dén tɛ",            "rule6_restrictive"),
    ("il n'est qu'un lâche",         "a tɛ foyi yé ni sègɛ tɛ",           "rule6_restrictive"),
    ("elle n'est que belle",         "a tɛ foyi yé ni ɲuman tɛ",          "rule6_restrictive_adj"),

    # Rule 7 : Qu'est-ce que / contenu expletif (variantes)
    ("Qu'est-ce qu'il a fait ?",     "a yé mún kɛ ?",                     "content_question_past"),
    ("Qu'est-ce que vous voulez ?",  "mún aw bɛ ŋàniya ?",                "rule7_quest_ce_que_plur"),

    # Rule 8 : relative + construction complexe (variantes)
    ("L'homme que tu vois est mon ami","i bɛ cɛ mìn yé, o yé n terikɛ yé", "rule8_relative_topic"),

    # Rule 9 : valoir la peine (variantes)
    ("cela ne vaut pas la peine",    "o tɛ buntu cɛ wà",                  "rule9_valoir_peine_neg"),

    # ════════════════════════════════════════════════════════════════════
    # XLIX. EXPERIENCER STATE CONSTRUCTION — avoir + état subjectif
    # Bambara : STATE bɛ SUBJ la  /  SUBJ_POSS CORPS bɛ SUBJ PAIN
    # ════════════════════════════════════════════════════════════════════

    ("J'ai faim",                    "kɔngɔ bɛ n la",                     "experiencer_faim"),
    ("J'ai soif",                    "jáabi bɛ n la",                     "experiencer_soif"),
    ("il a peur",                    "siran bɛ a la",                     "experiencer_peur"),
    ("elle a honte",                 "maloya bɛ a la",                    "experiencer_honte"),
    ("j'ai sommeil",                 "sùn bɛ n la",                       "experiencer_sommeil"),
    ("nous avons froid",             "sɛgi bɛ an la",                     "experiencer_froid"),
    ("tu as chaud",                  "teliman bɛ i la",                   "experiencer_chaud"),
    ("je n'ai pas faim",             "kɔngɔ tɛ n la",                     "experiencer_faim_neg"),
    ("il n'a pas peur",              "siran tɛ a la",                     "experiencer_peur_neg"),

    # Cas douleur + partie du corps
    ("j'ai mal à la tête",           "n horon bɛ n dimi",                 "experiencer_pain_tete"),
    ("elle a mal au ventre",         "a kɔnɔ bɛ a dimi",                  "experiencer_pain_ventre"),
    ("il a mal aux pieds",           "a sew bɛ a dimi",                   "experiencer_pain_pied"),

    # ════════════════════════════════════════════════════════════════════
    # L. FRÉQUENCE ET DISTRIBUTION TEMPORELLE
    # Bambara : distributif sɛbɛ / tɛmɛnen kɔrɔ / lɔgɔ kelen kelen…
    # ════════════════════════════════════════════════════════════════════

    ("tous les jours",               "dón bɛɛ",                           "freq_tous_les_jours"),
    ("chaque jour",                  "dón bɛɛ",                           "freq_chaque_jour"),
    ("chaque mois",                  "kalo kelen kelen",                  "freq_chaque_mois"),
    ("chaque semaine",               "dɔgɔkun kelen kelen",               "freq_chaque_semaine"),
    ("chaque année",                 "san kelen kelen",                   "freq_chaque_annee"),
    ("tous les matins",              "sɔgɔma bɛɛ",                        "freq_tous_les_matins"),
    ("tous les soirs",               "wùlafɛ bɛɛ",                        "freq_tous_les_soirs"),
    ("il vient tous les jours",      "a bɛ nà dón bɛɛ",                   "freq_vient_tous_les_jours"),
    ("elle mange chaque matin",      "a bɛ dún sɔgɔma bɛɛ",               "freq_mange_chaque_matin"),
    ("nous travaillons chaque semaine", "an bɛ báara kɛ dɔgɔkun kelen kelen", "freq_travaille_chaque_semaine"),

    # ════════════════════════════════════════════════════════════════════
    # M. PHRASES IMPERSONNELLES
    # Météorologiques (phénomène bɛ na), obligation (a ka kan ka V),
    # condition météo (a bɛ X la)
    # ════════════════════════════════════════════════════════════════════

    # Météorologiques
    ("il pleut",                     "fùrufuru bɛ na",                    "impersonnel_meteo"),
    ("il neige",                     "nínaban bɛ na",                     "impersonnel_meteo"),
    ("il fait chaud",                "a bɛ híjira la",                    "impersonnel_meteo"),
    ("il fait froid",                "a bɛ bùnaki la",                    "impersonnel_meteo"),
    ("il fait nuit",                 "súfɛla bɛ",                         "impersonnel_meteo"),

    # Obligation / nécessité (il faut / il ne faut pas)
    ("il faut manger",               "a ka kan ka dún",                   "impersonnel_falloir"),
    ("il faut partir",               "a ka kan ka táa",                   "impersonnel_falloir"),
    ("il ne faut pas manger",        "a man kan ka dún",                  "impersonnel_falloir_neg"),
    ("il ne faut pas partir",        "a man kan ka táa",                  "impersonnel_falloir_neg"),

    # Devoir / obligation personnelle
    ("il doit partir",               "a ka kan ka táa",                   "impersonnel_devoir"),
]


# ── POS TREE EXTRACTION ────────────────────────────────────────────────────────
def get_french_pos_tree(phrase, tagger=None):
    """Extract POS sequence from French phrase for syntactic tree comparison."""
    if not tagger:
        return ""
    try:
        tokens = tagger.tag_and_parse(phrase)
        pos_list = [t.get('pos', 'X') for t in tokens]
        return ' '.join(pos_list)
    except:
        return ""

def get_bambara_pos_tree(phrase_fr, bambara_result, engine=None, tagger=None):
    """
    Extract POS sequence from Bambara translation.
    Uses heuristic classification based on known Bambara markers and word patterns.
    """
    if not bambara_result or not tagger:
        return ""
    try:
        # Tokenize Bambara output by whitespace and punctuation
        import re
        # Keep punctuation separate
        bambara_result_clean = bambara_result.replace('?', ' ?').replace(',', ' ,')
        words = bambara_result_clean.split()

        # Known Bambara markers and their POS classes
        TAM_MARKERS = {'bɛ', 'yé', 'ma', 'tɛ', 'tùn', 'kà', 'ka', 'ka tɛ', 'na', 'ra'}
        PREPOSITIONS = {'la', 'kɔnɔ', 'kɛ', 'ni', 'le', 'te', 'den', 'min', 'don', 'dòn', 'ko', 'ní', 'mána'}
        PRONOUNS = {'n', 'i', 'a', 'o', 'u', 'anw', 'aw', 'ùw', 'inw', 'iw'}
        CONJUNCTIONS = {'ka', 'o', 'wà', 'dun', 'foyi', 'yé', 'ni', 'ni', 'te'}

        pos_list = []
        for word in words:
            word_clean = word.strip('.,?;:')
            if not word_clean:
                continue

            # Classify based on known patterns
            if word_clean in TAM_MARKERS:
                pos_list.append('AUX')  # TAM markers are auxiliary
            elif word_clean in PRONOUNS:
                pos_list.append('PRON')
            elif word_clean in PREPOSITIONS:
                pos_list.append('ADP')  # Adposition/preposition
            elif word_clean in CONJUNCTIONS:
                pos_list.append('CCONJ')  # Coordinating conjunction
            elif word_clean == '?':
                pos_list.append('PUNCT')
            elif word_clean == ',':
                pos_list.append('PUNCT')
            elif word_clean[0].isupper():  # Proper noun
                pos_list.append('PROPN')
            elif word_clean.endswith('li') or word_clean.endswith('ra') or word_clean.endswith('na'):
                pos_list.append('VERB')  # Verb forms
            else:
                # Default classification based on context
                # Most common in Bambara after TAM: nouns, adjectives, verbs
                pos_list.append('NOUN')  # Default to noun

        return ' '.join(pos_list) if pos_list else ""
    except:
        return ""


# ── CAMEMBERT SEMANTIC SIMILARITY ─────────────────────────────────────────────
_camembert_model = None

def _load_camembert():
    global _camembert_model
    if _camembert_model is None:
        from sentence_transformers import SentenceTransformer
        print("  [CamemBERT] Chargement du modèle dangvantuan/sentence-camembert-large …")
        _camembert_model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    return _camembert_model


def backtranslate_bm_to_fr(bm_text: str) -> str:
    """Back-translate Bambara → French via Google Translate."""
    if not bm_text:
        return ''
    try:
        from googletrans import Translator
        r = Translator().translate(bm_text, src='bm', dest='fr')
        return r.text or ''
    except Exception:
        return ''


def compute_camembert_scores(phrases_fr: list, hyps_bm: list):
    """
    Back-translate hyps_bm (Bambara) → French, then compute cosine similarity
    with the original French source using CamemBERT embeddings.

    Returns (scores: list[float], back_translations: list[str])
    """
    import numpy as np
    model = _load_camembert()

    back_fr = [backtranslate_bm_to_fr(h) for h in hyps_bm]

    src_emb = model.encode(phrases_fr, convert_to_numpy=True, normalize_embeddings=True)
    hyp_emb = model.encode(
        [h if h else ' ' for h in back_fr],
        convert_to_numpy=True, normalize_embeddings=True,
    )
    scores = [float(np.dot(s, h)) for s, h in zip(src_emb, hyp_emb)]
    return scores, back_fr


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
                result = translate_fn(phrase).strip()
                ok = result == expected.strip()

                # Extract POS trees
                fr_pos = get_french_pos_tree(phrase, tagger)
                bm_pos = get_bambara_pos_tree(phrase, result, engine=None, tagger=tagger)

                if ok:
                    passed += 1
                    print(f"  ✅  : {result}")
                    results.append((i, category, phrase, expected, result, 'PASS',
                                    fr_pos, bm_pos, google_bm, nllb_bm))
                else:
                    failed.append((i, category, phrase, expected, result))
                    print(f"  ❌  : {result}")
                    results.append((i, category, phrase, expected, result, 'FAIL',
                                    fr_pos, bm_pos, google_bm, nllb_bm))
            except Exception as e:
                err = f"ERREUR: {e}"
                failed.append((i, category, phrase, expected, err))
                print(f"  💥  : {e}")
                results.append((i, category, phrase, expected, err, 'ERROR',
                                '', '', google_bm, nllb_bm))
        else:
            results.append((i, category, phrase, expected, '', '', '', '',
                            google_bm, nllb_bm))
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

    # ── CamemBERT semantic similarity (batch) ─────────────────────────
    cam_kuma   = [''] * len(results)
    cam_google = [''] * len(results)
    cam_nllb   = [''] * len(results)

    if translate_fn and not no_camembert:
        try:
            print("  [CamemBERT] Back-translation BM→FR + calcul similarité …")
            _phrases = [r[2] for r in results]
            _kuma_bm  = [r[4] if r[5] != 'ERROR' else '' for r in results]
            _goog_bm  = [r[8] for r in results]
            _nllb_bm  = [r[9] for r in results]

            _sc_k, _ = compute_camembert_scores(_phrases, _kuma_bm)
            _sc_g, _ = compute_camembert_scores(_phrases, _goog_bm)
            _sc_n, _ = compute_camembert_scores(_phrases, _nllb_bm)

            cam_kuma   = [f'{s:.3f}' for s in _sc_k]
            cam_google = [f'{s:.3f}' for s in _sc_g]
            cam_nllb   = [f'{s:.3f}' for s in _sc_n]

            avg_k = sum(_sc_k) / len(_sc_k)
            avg_g = sum(_sc_g) / len(_sc_g)
            avg_n = sum(_sc_n) / len(_sc_n)
            print(f"\n  ── CamemBERT (similarité FR après back-trad BM→FR) ──")
            print(f"  Kuma-MT : {avg_k:.3f}")
            print(f"  Google  : {avg_g:.3f}")
            print(f"  NLLB    : {avg_n:.3f}\n")
        except Exception as _e:
            print(f"  [CamemBERT] Erreur : {_e} — colonnes laissées vides")

    # ── Export CSV ─────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"resultats_bambara_{ts}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '#', 'categorie', 'phrase_fr',
            'bambara_attendu', 'bambara_obtenu', 'statut',
            'french_pos_tree', 'bambara_pos_tree',
            'google_translate', 'nllb_translate',
            'camembert_kuma', 'camembert_google', 'camembert_nllb',
        ])
        for row, ck, cg, cn in zip(results, cam_kuma, cam_google, cam_nllb):
            writer.writerow(list(row) + [ck, cg, cn])
    _has_baseline = bool(_google_cache or _nllb_cache)
    print(f"  CSV exporté : {csv_path}")
    print(f"  Colonnes : bambara_obtenu | google_translate | nllb_translate"
          + (" (cache baseline chargé)" if _has_baseline else " (cache baseline absent)"))
    if translate_fn and not no_camembert:
        print(f"           | camembert_kuma | camembert_google | camembert_nllb")
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
            run_tests(translate_fn=lambda s: engine.translate(s)['bambara'],
                      no_camembert=_no_cam)
        except ImportError as e:
            print(f"Moteur non trouvé ({e})\n")
            run_tests(no_camembert=_no_cam)
        except Exception as e:
            print(f"Erreur d'initialisation : {e}\n")
            run_tests(no_camembert=_no_cam)
    else:
        run_tests(no_camembert=_no_cam)