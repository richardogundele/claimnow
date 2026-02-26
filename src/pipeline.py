# Main processing pipeline orchestrator
# Coordinates the full claim analysis workflow:
# S3 upload -> Textract -> Extractor -> Bedrock (if needed) -> Rate Matcher -> Scorer -> Explainer
