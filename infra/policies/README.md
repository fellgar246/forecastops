# Policies

Workload policies live in this directory and are attached by the IAM module.

`s3-deny-insecure-transport.json` uses a wildcard principal and `s3:*` because a transport deny has to cover every caller and every object operation. The statement is a deny, and it applies only to the project bucket that renders the template. Other policies name exact actions and resource ARNs.
