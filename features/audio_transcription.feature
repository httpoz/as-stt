Feature: Transcribing audio chunks with OpenAI
  As a creator preparing transcripts
  I want to upload the first chunk of a recording to OpenAI
  So that I can store the resulting transcription as plain text

  Scenario: Transcribing a user-specified chunk
    Given chunked files exist at "recording_chunk000.m4a" in the workspace root
    And the "OPENAI_API_KEY" environment variable is configured
    And "recording_chunk000.m4a" is smaller than 25 megabytes
    When I run `audio_splitter_cli transcribe recording_chunk000.m4a`
    Then the CLI validates that the chunk is no larger than 25 megabytes
    And it uploads the chunk to the "gpt-4o-transcribe" model
    And the returned transcript text is written to "recording_chunk000.m4a.txt"
    And I see a confirmation that the transcript file was saved
