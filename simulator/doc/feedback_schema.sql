BEGIN;

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS feedback_tariff_simulator (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email varchar(255) NOT NULL,
    company_name varchar(255) NOT NULL,
    comment text NOT NULL,
    attachments jsonb NULL,
    context jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_tariff_simulator_created_at
    ON feedback_tariff_simulator (created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_tariff_simulator_email
    ON feedback_tariff_simulator (email);

CREATE TABLE IF NOT EXISTS feedback_tariff_simulator_agree (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agree boolean NOT NULL,
    comment text NOT NULL,
    context jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_tariff_simulator_agree_created_at
    ON feedback_tariff_simulator_agree (created_at);

CREATE TABLE IF NOT EXISTS feedback_classification_issue (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email varchar(255) NOT NULL,
    company_name varchar(255) NOT NULL,
    comment text NOT NULL,
    expect_hts varchar(64) NOT NULL,
    agent_suggestion text NOT NULL,
    attachments jsonb NULL,
    classification_id varchar(128) NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_classification_issue_classification_id
    ON feedback_classification_issue (classification_id);
CREATE INDEX IF NOT EXISTS idx_feedback_classification_issue_created_at
    ON feedback_classification_issue (created_at);

CREATE TABLE IF NOT EXISTS feedback_classification_rating (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    classification_id varchar(128) NOT NULL,
    answerability smallint NOT NULL CHECK (answerability BETWEEN 1 AND 5),
    non_redundancy smallint NOT NULL CHECK (non_redundancy BETWEEN 1 AND 5),
    category_relevance smallint NOT NULL CHECK (category_relevance BETWEEN 1 AND 5),
    clarity smallint NOT NULL CHECK (clarity BETWEEN 1 AND 5),
    reasoning_quality smallint NOT NULL CHECK (reasoning_quality BETWEEN 1 AND 5),
    ux_score smallint NOT NULL CHECK (ux_score BETWEEN 1 AND 5),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_classification_rating_classification_id
    ON feedback_classification_rating (classification_id);
CREATE INDEX IF NOT EXISTS idx_feedback_classification_rating_created_at
    ON feedback_classification_rating (created_at);

CREATE TABLE IF NOT EXISTS feedback_feature_proposal (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title varchar(255) NOT NULL,
    reason text NOT NULL,
    current_workaround text NULL,
    frequency varchar(64) NULL,
    attachments jsonb NULL,
    user_role varchar(64) NOT NULL,
    company_size varchar(32) NULL,
    email varchar(255) NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_feature_proposal_created_at
    ON feedback_feature_proposal (created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_feature_proposal_user_role
    ON feedback_feature_proposal (user_role);

COMMIT;
