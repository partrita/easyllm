## 2024-05-18 - Exposed HF Credential
**Vulnerability:** An authentication secret was accidentally included in `notebooks/datasets/filter-dataset.ipynb` within a `push_to_hub` method call.
**Learning:** Hardcoding credentials often occurs in Jupyter Notebooks during testing or prototyping and accidentally gets committed. Notebooks are particularly susceptible to this because they combine code, output, and execution state.
**Prevention:** Always use environment variables (e.g., `os.environ.get("HF_TOKEN_ENV_VAR")`) or secure credential managers even in development notebooks. Implement pre-commit hooks (like `detect-secrets` or `trufflehog`) to scan notebooks before they are committed to version control.## 2025-04-01 - Prevent Sensitive User Prompt Data Exposure in Warning Logs
**Vulnerability:** The default prompt builder fallback logged the complete user prompt (`buildBasePrompt(request.messages)`) into standard application logs via `logger.warn`.
**Learning:** Automatically including dynamic user-supplied data (such as LLM prompts, which often contain PII or secrets) inside warning logs creates a critical data exposure risk for all system users relying on default library settings.
**Prevention:** Hardcode static error or warning messages that instruct the user to configure custom settings, strictly avoiding the interpolation of raw, unredacted user inputs into standard, non-debug application logs.

## 2024-05-24 - Prevent Sensitive API Response Data Exposure in Error Logs
**Vulnerability:** The SageMaker client's error handling logged the complete raw response body (`res.text`) into standard application logs via `logger.error` when the API returned a non-200 status code.
**Learning:** Automatically including dynamic external API responses inside error logs creates a critical data exposure risk, as these responses could contain raw sensitive data, internal stack traces, or even credentials that are unexpectedly returned by the server on error.
**Prevention:** Hardcode static error or warning messages that instruct the user about the failure status, strictly avoiding the interpolation of raw, unredacted external API responses into standard application logs.
