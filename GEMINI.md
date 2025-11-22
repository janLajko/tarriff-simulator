# Project Overview

This project is a tariff simulation API built with Python, FastAPI, and PostgreSQL. It provides an endpoint to simulate tariffs for imported goods based on their HTS (Harmonized Tariff Schedule) code, country of origin, and other factors.

The system is composed of two main parts:

1.  **Tariff Simulation API (`simulator/`):** A FastAPI application that exposes a `/simulate` endpoint. This endpoint takes details about an import (HTS code, country, value, etc.) and calculates the estimated tariff duty. The calculation is modular, with separate logic for:
    *   Basic HTS duty
    *   Section 301 tariffs
    *   Section 232 tariffs
    *   IEEPA (International Emergency Economic Powers Act) tariffs

    The API reads tariff data and rules from a PostgreSQL database. It also appears to use some JSON configuration files for HTS unit and formula information (`hts_unit.json`, `spi.json`).

2.  **Data Ingestion Agents (`agent/`):** A collection of Python scripts responsible for populating the PostgreSQL database. For example, the `basic-hts-agent` reads HTS data from CSV files, normalizes it, and loads it into the `hts_codes` table. These agents are the data pipeline that feeds the simulation API.

# Building and Running

## Prerequisites

*   Python 3.8+
*   PostgreSQL
*   Docker (optional, for running in a container)

## Running the API

1.  **Set up the database:**
    *   Make sure you have a PostgreSQL server running.
    *   Set the `DATABASE_DSN` environment variable to your PostgreSQL connection string (e.g., `postgresql://user:pass@localhost:5432/dbname`).

2.  **Populate the database:**
    *   Run the data ingestion agents to populate the database. For example, to load the basic HTS data:
        ```bash
        python agent/basic-hts-agent/basic_hts_agent.py --dsn $DATABASE_DSN
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the API server:**
    ```bash
    uvicorn simulator.app:app --host 0.0.0.0 --port 8080
    ```

## Running with Docker

The project includes a `Dockerfile` for containerizing the API.

1.  **Build the Docker image:**
    ```bash
    docker build -t tariff-simulator .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8080:8080 -e DATABASE_DSN="your_database_dsn" tariff-simulator
    ```

# Development Conventions

*   The project uses FastAPI for the API framework and Pydantic for data validation.
*   The business logic for tariff calculations is separated into different modules (`basic_hts_rate.py`, `section301_rate.py`, etc.).
*   The data ingestion scripts are located in the `agent/` directory.
*   The project uses `psycopg2` for connecting to the PostgreSQL database.
*   Tests are located in the `simulator/test/` directory and can be run with `pytest`.
