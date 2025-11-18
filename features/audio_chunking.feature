Feature: Audio chunking for OpenAI STT uploads
  As a creator preparing transcripts
  I want to split large recordings into files that satisfy the 25 MB Whisper limit
  So that I can automate speech-to-text without manual editing

  Scenario: Creating sequential 25 MB chunks
    Given a source audio file larger than 25 megabytes
    When I run `audio_splitter_cli chunk recording.m4a --max-size 25MB`
    Then the CLI emits sequential files no larger than 25 megabytes each
    And each chunk filename includes the original name and a numeric index
    And the final chunk is allowed to be smaller than 25 megabytes so the entire recording is preserved

  Scenario: Rejecting inputs that are already below the threshold
    Given a source audio file smaller than 25 megabytes
    When I run `audio_splitter_cli chunk short.m4a --max-size 25MB`
    Then the CLI exits with status code 0
    And it returns a single chunk identical to the input file
