QUANT_ALIASES = {
    "UD-IQ1_M": "IQ1_M",
    "UD-IQ2_XXS": "IQ2_XXS",
    "UD-IQ2_XSS": "IQ2_XSS",
    "UD-IQ2_M": "IQ2_M",
    "UD-Q2_K_XL": "Q2_K_XL",
    "UD-IQ3_XXS": "IQ3_XXS",
    "UD-IQ3_S": "IQ3_S",
    "UD-Q3_K_S": "Q3_K_S",
    "UD-Q3_K_M": "Q3_K_M",
    "UD-Q3_K_XL": "Q3_K_XL",
    "UD-IQ4_XS": "IQ4_XS",
    "UD-IQ4_NL": "IQ4_NL",
    "UD-IQ4_NL_XL": "IQ4_NL_XL",
    "UD-Q4_K_S": "Q4_K_S",
    "UD-Q4_K_M": "Q4_K_M",
    "UD-Q4_K_XL": "Q4_K_XL",
    "UD-Q5_K_S": "Q5_K_S",
    "UD-Q5_K_M": "Q5_K_M",
    "UD-Q5_K_XL": "Q5_K_XL",
    "UD-Q6_K": "Q6_K",
    "UD-Q6_K_XL": "Q6_K_XL",
    "UD-Q8_K_XL": "Q8_K_XL",
}

QUANT_POSITION = {
    "IQ1_M": 1.0,
    "IQ2_XXS": 2.0,
    "IQ2_XSS": 2.1,
    "IQ2_XS": 2.2,
    "IQ2_S": 2.3,
    "IQ2_M": 2.4,
    "Q2_K": 2.5,
    "Q2_K_L": 2.6,
    "Q2_K_XL": 2.8,
    "IQ3_XXS": 3.0,
    "IQ3_XS": 3.1,
    "IQ3_S": 3.2,
    "IQ3_M": 3.3,
    "Q3_K_S": 3.35,
    "Q3_K_M": 3.5,
    "Q3_K_L": 3.6,
    "Q3_K_XL": 3.7,
    "IQ4_XS": 3.9,
    "IQ4_NL": 3.95,
    "IQ4_NL_XL": 4.0,
    "MXFP4": 4.0,
    "MXFP4_MOE": 4.0,
    "MXFP4_MOE_BF16": 4.0,
    "MXFP4_MOE_F16": 4.0,
    "Q4_0": 4.05,
    "Q4_1": 4.1,
    "Q4_K_S": 4.15,
    "Q4_K_M": 4.35,
    "Q4_K_L": 4.45,
    "Q4_K_XL": 4.55,
    "Q5_K_S": 5.1,
    "Q5_K_M": 5.3,
    "Q5_K_L": 5.4,
    "Q5_K_XL": 5.5,
    "Q6_K_S": 6.0,
    "Q6_K": 6.1,
    "Q6_K_L": 6.2,
    "Q6_K_XL": 6.3,
    "Q8_0": 8.0,
    "Q8_K_XL": 8.1,
}


def canonical_quant(quant: str) -> str:
    return QUANT_ALIASES.get(quant, quant)


QUANT_ORDER = {
    quant: int(position * 100)
    for quant, position in {
        **QUANT_POSITION,
        **{alias: QUANT_POSITION[canonical] for alias, canonical in QUANT_ALIASES.items()},
    }.items()
}

UNKNOWN_QUANT_ORDER = max(QUANT_ORDER.values()) + 1
