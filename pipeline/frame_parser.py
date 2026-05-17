"""
pipeline/frame_parser.py
Detects dominant semantic frame by voting over Sense.frame in the KG.
Accepts a list of plain token strings (not tuples).
"""

from utils.normalize import normalize_token


class FrameParser:

    def __init__(self, db):
        self.db = db

    def detect_frame(self, tokens: list) -> str:
        """
        tokens: list of plain strings (lemmas).
        Filtre dynamiquement les valeurs nulles (None) pour éviter les crashs de normalisation.
        """
        if not tokens:
            return 'GENERIC'

        frame_votes = {}

        # LOGIQUE DATA-DRIVEN : On élimine les valeurs None ou vides avant traitement
        valid_tokens = [t for t in tokens if t is not None]

        for t in valid_tokens:
            # Sécurisation de l'extraction de la chaîne à normaliser
            if isinstance(t, str):
                norm = normalize_token(t)
            else:
                norm = normalize_token(t[0] if isinstance(t, (list, tuple)) and t else t)
                
            if not norm:
                continue

            res = self.db.query("""
            MATCH (w:Word {text:$t})-[:HAS_SENSE]->(s:Sense)
            WHERE s.frame IS NOT NULL
            RETURN s.frame AS frame LIMIT 5
            """, {"t": norm})

            if not res:
                res = self.db.query("""
                MATCH (w:Word)-[:HAS_SENSE]->(s:Sense)
                WHERE toLower(w.text) = $t AND s.frame IS NOT NULL
                RETURN s.frame AS frame LIMIT 5
                """, {"t": norm})

            # Parcours et comptabilisation des votes
            if res and isinstance(res, list):
                for row in res:
                    f = row.get('frame') or 'GENERIC'
                    frame_votes[f] = frame_votes.get(f, 0) + 1

        if not frame_votes:
            return 'GENERIC'
            
        return max(frame_votes.items(), key=lambda x: x[1])[0]
