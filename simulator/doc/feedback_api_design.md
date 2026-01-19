# Feedback API Design

## Overview

- Base URL: `/api/v1/feedback`
- Content-Type: `application/json`
- Auth: none
- Response format: `code` + `message`

Example response:
```json
{
  "code": "OK",
  "message": "success"
}
```

## Error Codes

| HTTP | Code | Description |
| --- | --- | --- |
| 400 | BAD_REQUEST | Invalid request parameters (e.g., rating out of 1-5) |
| 500 | SERVER_ERROR | Internal server error |

## Attachment Upload (GCS Direct Upload)

Uploads the file to GCS and returns the object URL for use in `attachments`.

- Bucket: `aitryon-images`
- Object key format: `feedback/{yyyy}/{mm}/{uuid}_{filename}`
- Max size: 10MB

Endpoint:
`POST /api/v1/feedback/attachments/upload`

Request:
`multipart/form-data`

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| file | file | yes | Binary file |

Response:
```json
{
  "code": "OK",
  "message": "success",
  "data": {
    "object_url": "https://storage.googleapis.com/aitryon-images/feedback/2025/01/uuid_filename.png",
    "gcs_uri": "gs://aitryon-images/feedback/2025/01/uuid_filename.png",
    "object_name": "feedback/2025/01/uuid_filename.png",
    "content_type": "image/png",
    "size_bytes": 12345
  }
}
```

## Tariff Simulator Feedback

Endpoint:
`POST /api/v1/feedback/tariff-simulator/feedback`

Request Body:
| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| email | string | yes | User contact email |
| company_name | string | yes | Company name |
| comment | string | yes | Feedback content |
| attachments | array[string] | no | GCS `object_url` list |
| context | object | yes | Arbitrary JSON object |

Example:
```json
{
  "email": "user@example.com",
  "company_name": "ACME Inc.",
  "comment": "Metal ratio seems incorrect",
  "attachments": [
    "https://storage.googleapis.com/aitryon-images/feedback/2025/01/uuid_file.png"
  ],
  "context": {
    "request": {"hts":"0101.01.0000","origin_country": "CN"},
    "response": {"rate":0.1,"origin_country": "CN"}
  }
}
```

## Tariff Simulator Agree/Disagree

Endpoint:
`POST /api/v1/feedback/tariff-simulator/agree`

Request Body:
| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| agree | boolean | yes | Agree or disagree |
| context | object | yes | Arbitrary JSON object |
| comment | string | no | Feedback content |

Example:
```json
{
  "comment": "Result looks right",
  "context": {
    "request": {"hts":"0101.01.0000","origin_country": "CN"},
    "response": {"rate":0.1,"origin_country": "CN"}
  },
  "agree": true
}
```

## Classification Issue Reporting

Endpoint:
`POST /api/v1/feedback/classification/issue`

Request Body:
| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| email | string | yes | User contact email |
| company_name | string | yes | Company name |
| comment | string | yes | Error description or improvement suggestion |
| expect_hts | string | yes | Expected HTS |
| agent_suggestion | string | yes | Chat log + audit.log (client provided) |
| attachments | array[string] | no | Screenshot URLs |
| classification_id | string | no | Classification task ID |

Example:
```json
{
  "email": "user@example.com",
  "company_name": "ACME Inc.",
  "comment": "分类结果偏高",
  "expect_hts": "9903.88.03",
  "agent_suggestion": "chat-log... audit.log...",
  "attachments": [
    "https://storage.googleapis.com/aitryon-images/feedback/2025/01/uuid_file.png"
  ],
  "classification_id": "cls_123"
}
```

## Classification Satisfaction Rating

Endpoint:
`POST /api/v1/feedback/classification/rating`

Request Body (all ratings are integers 1-5):
| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| answerability | integer | yes | Answerability |
| non_redundancy | integer | yes | Non-redundancy |
| category_relevance | integer | yes | Category relevance |
| clarity | integer | yes | Clarity |
| reasoning_quality | integer | yes | Reasoning quality |
| ux_score | integer | yes | UX |
| classification_id | string | yes | Classification task ID |

Example:
```json
{
  "answerability": 4,
  "non_redundancy": 5,
  "category_relevance": 4,
  "clarity": 4,
  "reasoning_quality": 3,
  "ux_score": 4,
  "classification_id": "cls_123"
}
```

## Feature Proposal

Endpoint:
`POST /api/v1/feedback/feature-proposal`

Request Body:
| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| title | string | yes | Feature title |
| reason | string | yes | Why this is needed |
| current_workaround | string | no | Current workaround |
| frequency | string | no | Usage frequency |
| attachments | array[string] | no | Image URLs |
| user_role | string | yes | User role (string) |
| company_size | string | no | Company size |
| email | string | no | Contact email |

Example:
```json
{
  "title": "批量导入 HTS",
  "reason": "当前只能单条查询，效率低",
  "current_workaround": "用脚本调用接口",
  "frequency": "每周",
  "attachments": [],
  "user_role": "procurement",
  "company_size": "11-50",
  "email": "user@example.com"
}
```
