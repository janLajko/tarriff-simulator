# Feedback Database Design

## Overview

- Database: PostgreSQL
- Timestamps: `created_at` stored in UTC (`timestamptz`)
- Attachments stored as JSON arrays of GCS object URLs
- Context stored as JSON object with arbitrary keys

## Tables

### 1) feedback_tariff_simulator

| Column | Type | Constraints | Notes |
| --- | --- | --- | --- |
| id | uuid | PK | Feedback ID |
| email | varchar(255) | NOT NULL | User email |
| company_name | varchar(255) | NOT NULL | Company name |
| comment | text | NOT NULL | Feedback content |
| attachments | jsonb | NULL | URL array |
| context | jsonb | NOT NULL | Arbitrary JSON object |
| created_at | timestamptz | NOT NULL | Created time |

Indexes:
- `created_at`
- `email`

### 2) feedback_tariff_simulator_agree

| Column | Type | Constraints | Notes |
| --- | --- | --- | --- |
| id | uuid | PK | Record ID |
| agree | boolean | NOT NULL | Agree/disagree |
| comment | text | NOT NULL | Feedback content |
| context | jsonb | NOT NULL | Request/response context |
| created_at | timestamptz | NOT NULL | Created time |

Indexes:
- `created_at`

### 3) feedback_classification_issue

| Column | Type | Constraints | Notes |
| --- | --- | --- | --- |
| id | uuid | PK | Issue ID |
| email | varchar(255) | NOT NULL | User email |
| company_name | varchar(255) | NOT NULL | Company name |
| comment | text | NOT NULL | Error description |
| expect_hts | varchar(64) | NOT NULL | Expected HTS |
| agent_suggestion | text | NOT NULL | Chat log + audit.log |
| attachments | jsonb | NULL | URL array |
| classification_id | varchar(128) | NULL | Classification task ID |
| created_at | timestamptz | NOT NULL | Created time |

Indexes:
- `classification_id`
- `created_at`

### 4) feedback_classification_rating

| Column | Type | Constraints | Notes |
| --- | --- | --- | --- |
| id | uuid | PK | Rating ID |
| classification_id | varchar(128) | NOT NULL | Classification task ID |
| answerability | smallint | CHECK 1-5 | Answerability |
| non_redundancy | smallint | CHECK 1-5 | Non-redundancy |
| category_relevance | smallint | CHECK 1-5 | Category relevance |
| clarity | smallint | CHECK 1-5 | Clarity |
| reasoning_quality | smallint | CHECK 1-5 | Reasoning quality |
| ux_score | smallint | CHECK 1-5 | UX |
| created_at | timestamptz | NOT NULL | Created time |

Indexes:
- `classification_id`
- `created_at`

### 5) feedback_feature_proposal

| Column | Type | Constraints | Notes |
| --- | --- | --- | --- |
| id | uuid | PK | Proposal ID |
| title | varchar(255) | NOT NULL | Feature title |
| reason | text | NOT NULL | Why it is needed |
| current_workaround | text | NULL | Current workaround |
| frequency | varchar(64) | NULL | Usage frequency |
| attachments | jsonb | NULL | URL array |
| user_role | varchar(64) | NOT NULL | User role |
| company_size | varchar(32) | NULL | Company size |
| email | varchar(255) | NULL | Contact email |
| created_at | timestamptz | NOT NULL | Created time |

Indexes:
- `created_at`
- `user_role`
