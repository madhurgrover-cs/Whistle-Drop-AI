"""Aggregates every v1 router into a single mountable router.

Keeping one aggregation point means ``main.py`` never grows a list of imports
as features are added — each phase registers its router here.
"""

from fastapi import APIRouter

from app.api.v1.routers import auth, cases, health, moderation, reports

api_router = APIRouter()

# Public, unauthenticated.
api_router.include_router(health.router)
api_router.include_router(reports.router)
api_router.include_router(cases.router)

# Moderator-facing. Login is public by necessity; everything under
# /moderation requires the bearer token it issues.
api_router.include_router(auth.router)
api_router.include_router(moderation.router)
