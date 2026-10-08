"""API Endpoint listing supported Indian Cattle and Buffalo breeds."""

from typing import Optional
from fastapi import APIRouter, Query
from app.schemas.predict import BreedsListResponseSchema, BreedInfoSchema
from ml.common.breed_registry import BreedRegistry

router = APIRouter(tags=["Breed Metadata"])

BREED_DESCRIPTIONS = {
    "Dangi": {
        "origin_region": "Maharashtra / Gujarat (Nashik, Ahmednagar, Dang)",
        "description": "Hardy indigenous draught zebu cattle breed recognized by distinctive black/white or red/white mottled coat and resilience to heavy rainfall.",
    },
    "Gaolao": {
        "origin_region": "Maharashtra / Madhya Pradesh (Wardha, Nagpur, Chhindwara)",
        "description": "Dual-purpose draught and milk cattle breed characterized by white or light grey coat, convex forehead, and agile build.",
    },
    "Gir": {
        "origin_region": "Gujarat (Gir Hills / Kathiawar)",
        "description": "Premier indigenous dairy cattle breed known for distinctive rounded forehead, long pendulous ears, and high milk yield.",
    },
    "Ongole": {
        "origin_region": "Andhra Pradesh (Prakasam District)",
        "description": "Famous dual-purpose draft and dairy cattle breed recognized by muscular hump, white coat, and high disease resistance.",
    },
    "Sahiwal": {
        "origin_region": "Punjab / Rajasthan (Indo-Pak Border Region)",
        "description": "Premier indigenous zebu dairy cattle breed known for reddish-brown coat, loose skin folds, and high milk production.",
    },
    "Manda": {
        "origin_region": "Odisha / Andhra Pradesh (Eastern Ghats / Koraput)",
        "description": "Hardy indigenous riverine/swamp buffalo breed of the Eastern Ghats characterized by ash-grey/copper coat and high draft resilience.",
    },
    "Surti": {
        "origin_region": "Gujarat (Surat / Kheda / Anand)",
        "description": "Medium-sized riverine buffalo breed known for sickle-shaped horns, compact body, and high fat percentage milk.",
    },
    "Toda": {
        "origin_region": "Tamil Nadu (Nilgiri Hills)",
        "description": "Unique indigenous semi-wild buffalo breed reared by the Toda pastoralist tribe, possessing long crescent horns and thick fawn-coloured hair.",
    },
}


@router.get(
    "/breeds",
    response_model=BreedsListResponseSchema,
    summary="Get List of Supported Indian Cattle and Buffalo Breeds",
    description="Returns metadata catalog for all supported indigenous Indian cattle and buffalo breeds.",
)
async def get_supported_breeds(
    version: Optional[str] = Query(None, description="Optional model version filter"),
) -> BreedsListResponseSchema:
    """Retrieve catalog of supported Indian cattle and buffalo breeds."""
    if version and "selected" in version.lower():
        selected_order = [
            ("Dangi", "Dangi Cattle", "cattle"),
            ("Gaolao", "Gaolao Cattle", "cattle"),
            ("Gir", "Gir Cattle", "cattle"),
            ("Ongole", "Ongole Cattle", "cattle"),
            ("Sahiwal", "Sahiwal Cattle", "cattle"),
            ("Manda", "Manda Buffalo", "buffalo"),
            ("Surti", "Surti Buffalo", "buffalo"),
            ("Toda", "Toda Buffalo", "buffalo"),
        ]
        selected_schemas = [
            BreedInfoSchema(
                breed_name=bname,
                display_name=dname,
                animal_type=atype,
                origin_region=BREED_DESCRIPTIONS[bname]["origin_region"],
                description=BREED_DESCRIPTIONS[bname]["description"],
            )
            for bname, dname, atype in selected_order
        ]
        return BreedsListResponseSchema(total_breeds=len(selected_schemas), breeds=selected_schemas)
    registry = BreedRegistry()
    all_breeds = registry.get_all_breeds(enabled_only=True)

    breed_schemas = []
    for info in all_breeds:
        meta = BREED_DESCRIPTIONS.get(
            info.breed_name,
            {
                "origin_region": "India",
                "description": f"Indigenous Indian {info.animal_type} breed.",
            },
        )

        breed_schemas.append(
            BreedInfoSchema(
                breed_name=info.breed_name,
                display_name=info.display_name,
                animal_type=info.animal_type,
                origin_region=meta["origin_region"],
                description=meta["description"],
            )
        )

    return BreedsListResponseSchema(
        total_breeds=len(breed_schemas), breeds=breed_schemas
    )
