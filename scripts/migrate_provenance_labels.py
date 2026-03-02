"""
Migration: add provenance_labels table to khipu_database.db.

Creates:
    provenance_labels(raw TEXT PRIMARY KEY, display_name TEXT NOT NULL)

Seeds it with the canonical set of friendly labels.  Safe to re-run — uses
INSERT OR REPLACE so existing rows are updated if the display name changes.

Usage:
    python scripts/migrate_provenance_labels.py
"""

import pathlib
import sqlite3

ROOT    = pathlib.Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "kfg" / "khipu_database.db"

# Full mapping of raw DB provenance strings → short display labels.
# Edit this table (not browse.py) to update or add labels.
LABELS: list[tuple[str, str]] = [
    # Chala group — DB stores these with surrounding double-quotes; both variants kept
    ('"This quipu is associated with AS59-AS67 / found with a cloth bag at Chala."', "Chala"),
    ('"This quipu is associated wtih AS59-AS67 / found with a cloth bag at Chala."', "Chala"),
    ('"This quipu, along with AS60-AS67, was found with a cloth bag at Chala."',     "Chala"),
    # Unquoted forms (belt-and-suspenders)
    ("This quipu is associated with AS59-AS67 / found with a cloth bag at Chala.",   "Chala"),
    ("This quipu is associated wtih AS59-AS67 / found with a cloth bag at Chala.",   "Chala"),
    ("This quipu, along with AS60-AS67, was found with a cloth bag at Chala.",       "Chala"),
    # Long Ascher museum-note
    ('Ascher notes "The museum card reads "From an Inca grave, Pachacamac, Peru. '
     'E. Nordenski\u00f6ld collection. By exchange 1925."',
     "Pachacamac (Nordenski\u00f6ld)"),
    # Site names
    ("Aankoop",                                                "Aankoop"),
    ("Acari",                                                  "Acari"),
    ("Armatambo, Huaca San Pedro",                             "Armatambo / Huaca San Pedro"),
    ("Armatambo, Lima, Central Coast",                         "Armatambo, Lima"),
    ("Atarco",                                                 "Atarco"),
    ("Between Ica and Pisco",                                  "Between Ica and Pisco"),
    ("Cajamarquilla",                                          "Cajamarquilla"),
    ("Casa del Quipu, Pachacamac",                             "Pachacamac (Casa del Quipu)"),
    ("Centinela, Tambe de Mora",                               "La Centinela / Tambo de Mora"),
    ("Chancay",                                                "Chancay"),
    ("Chancay, Central Coast",                                 "Chancay (Central Coast)"),
    ("Chuquitanta",                                            "Chuquitanta"),
    ("Cieneguilla, Valle de Lurin",                            "Cieneguilla (Lurin Valley)"),
    ("Costa Central, Huacho",                                  "Huacho (Central Coast)"),
    ("Costa Sur",                                              "South Coast"),
    ("Cuzco",                                                  "Cuzco"),
    ("Donation from the collection Belli",                     "Belli Collection"),
    ("Eduard Gaffron",                                         "Gaffron Collection"),
    ("Eduard Gaffron Estate",                                  "Gaffron Estate"),
    ("Grave K, road between Chulpaca and Tate (Site T), Ica Valley",
                                                               "Ica Valley (Site T, Grave K)"),
    ("Grave M, Site T, Ica; excavated by Max Uhle",            "Ica (Site T, Grave M \u2014 Uhle)"),
    ("Grave M, road between Chulpaca and Tate (Site T), Ica valley",
                                                               "Ica Valley (Site T, Grave M)"),
    ("Hacienda Copara, Nazca",                                 "Nazca (Hda. Copara)"),
    ("Hacienda Ullujalla y Callengo",                          "Hda. Ullujalla / Callengo"),
    ("Hda. Huando, Chancay",                                   "Chancay (Hda. Huando)"),
    ("Huaca Perez, Lima (a.k.a Hda. Infantas and Tambo Inca)", "Lima (Huaca P\u00e9rez)"),
    ("Huaca San Marco, possibly epoch 2 of the Middle Horizon period (AD 650\u2013750)",
                                                               "Huaca San Marco"),
    ("Huaca San Pedro, Armatambo",                             "Armatambo (Huaca San Pedro)"),
    ("Huacho",                                                 "Huacho"),
    ("Huacho u . Pachacamac",                                  "Huacho / Pachacamac"),
    ("Huacho?",                                                "Huacho (?)"),
    ("Huacones",                                               "Huacones"),
    ("Huando, Chancay, Peru (Gaffron Collection)",              "Chancay / Huando (Gaffron)"),
    ("Huaquerones",                                            "Huaquerones"),
    ("Huari",                                                  "Huari"),
    ("Ica",                                                    "Ica"),
    ("Ica Valley, near Callango",                              "Ica Valley (near Callango)"),
    ("Ica or Cajamarquilla",                                   "Ica / Cajamarquilla"),
    ("Ica, Coast of Peru",                                     "Ica (Coast)"),
    ("Ica/Pisco",                                              "Ica / Pisco"),
    ("Incahuasi",                                              "Incahuasi"),
    ("La Centinela,Tambo de Mora",                             "La Centinela / Tambo de Mora"),
    ("La Molina",                                              "La Molina"),
    ("La puntilla, between Paracas and Pisco",                 "La Puntilla (Paracas/Pisco)"),
    ("Leymebamba",                                             "Leymebamba"),
    ("Likely near Lima",                                       "Near Lima (prob.)"),
    ("Lima",                                                   "Lima"),
    ("Lluta Valley",                                           "Lluta Valley"),
    ("Maranga, Huaca 1",                                       "Lima (Maranga, Huaca 1)"),
    ("Marquez",                                                "Marquez"),
    ("Mollepampa",                                             "Mollepampa"),
    ("Monte de Cacatilla, Valle de Nazca",                     "Nazca (Monte de Cacatilla)"),
    ("Nazca",                                                  "Nazca"),
    ("Nazca Valley; Ancon, Central Coast",                     "Nazca / Ancon"),
    ("Ocucaje",                                                "Ocucaje"),
    ("Pacasmayo",                                              "Pacasmayo"),
    ("Pachacamac",                                             "Pachacamac"),
    ("Pachacamac (Casa de los quipus)",                        "Pachacamac (Casa de los Quipus)"),
    ("Paracas",                                                "Paracas"),
    ("Peru",                                                   "Peru (unknown)"),
    ("Peru, Fundort: Pachacmac",                               "Pachacamac (Fundort)"),
    ("Pisco",                                                  "Pisco"),
    ("Playa Miller #6, Arica, Chile",                          "Arica, Chile (Playa Miller 6)"),
    ("Probably collected by Jane Stanford and donated to the Stanford Museum before 1905",
                                                               "Stanford Collection (prob. 1905)"),
    ("Pueblo Libre, Lima, Peru",                               "Lima (Pueblo Libre)"),
    ("Purported to have been discovered in a burial at the coastal site of Ancon, near Lima, Peru",
                                                               "Ancon (prob.)"),
    ("Purucucho",                                              "Purucucho"),
    ("Rancho San Juan, Ica Valley, Peru",                      "Ica Valley (Rancho San Juan)"),
    ("Regi\u00f3n Sur, Quillagua, Valle de Loa",               "Quillagua, Valle de Loa"),
    ("Rimac Valley",                                           "Rimac Valley"),
    ("Santa",                                                  "Santa"),
    ("Santa Clara, Nazca",                                     "Nazca (Santa Clara)"),
    ("South Peru",                                             "South Peru"),
    ("Southern Coast, Peru",                                   "Southern Coast"),
    ("Tambo Colorado",                                         "Tambo Colorado"),
    ("Thomas Harper Goodspeed",                                "Goodspeed Collection"),
    ("Ullujaya, Ocucaje, Ica",                                 "Ocucaje / Ullujaya (Ica)"),
    ("Unknown",                                                "Unknown"),
    ("Unknown (not from Gaffron collections)",                 "Unknown (non-Gaffron)"),
    ("Valle de Ica Hacienda Callango Ocucaje",                 "Ica Valley (Hda. Callango / Ocucaje)"),
    ("Valle de Pisco",                                         "Pisco Valley"),
    ("foothills of Cerro Solar",                               "Cerro Solar (foothills)"),
    ("near Callengo, Ica Valley",                              "Ica Valley (near Callengo)"),
    ("near Lima",                                              "Near Lima"),
    ("probably Central Coast Late Period",                     "Central Coast (Late Period, prob.)"),
]


def run() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS provenance_labels (
            raw          TEXT PRIMARY KEY,
            display_name TEXT NOT NULL
        )
    """)

    cur.executemany(
        "INSERT OR REPLACE INTO provenance_labels (raw, display_name) VALUES (?, ?)",
        LABELS,
    )
    conn.commit()

    count = cur.execute("SELECT COUNT(*) FROM provenance_labels").fetchone()[0]
    print(f"provenance_labels table: {count} rows")
    conn.close()


if __name__ == "__main__":
    run()
