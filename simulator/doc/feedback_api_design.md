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
