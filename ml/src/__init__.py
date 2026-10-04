"""Offline training pipeline for WhistleDrop's advisory triage model.

Deliberately independent of ``app/``: nothing here imports FastAPI, opens a
database connection, or reads application settings. Training runs from a
checkout and a dataset file, and produces artifacts. Phase 7 is what loads
them.
"""
