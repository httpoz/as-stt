Feature: Audio chunking for OpenAI STT uploads
  As a creator preparing transcripts
  I want to split large recordings into files that satisfy the 25 MB Whisper limit
  So that I can automate speech-to-text without manual editing

  Scenario: Creating sequential 25 MB chunks
    Given a source audio file larger than 25 megabytes
    When I run `audio_splitter_cli chunk recording.m4a --max-size 25MB`
    And the CLI queries ffprobe for duration and bitrate metadata
    Then the CLI emits sequential files no larger than 25 megabytes each
    And each chunk filename includes the original name and a numeric index
    And the final chunk is allowed to be smaller than 25 megabytes so the entire recording is preserved
    And no chunk exceeds 1400 seconds so every chunk can be transcribed with buffer

  Scenario: Calculating chunk boundaries from metadata
    Given ffmpeg reports a duration of 3600 seconds and a bitrate of 228 kb/s
    When I calculate the chunk plan for a 25 megabyte limit
    Then I get chunk windows at 0s, 864s, 1728s, 2592s, and 3456s
    And the final chunk covers the remaining 144 seconds of audio

  Scenario: Enforcing a duration limit while chunking
    Given ffmpeg reports a duration of 4000 seconds and a bitrate of 128 kb/s
    And 25 megabytes of audio at that bitrate would exceed 1400 seconds
    When I calculate the chunk plan
    Then each chunk stops at or before the 1400 second buffer duration limit
    And the final chunk is smaller if needed so the entire recording is preserved

  Scenario: Rejecting inputs that are already below the threshold
    Given a source audio file smaller than 25 megabytes
    When I run `audio_splitter_cli chunk short.m4a --max-size 25MB`
    Then the CLI exits with status code 0
    And it returns a single chunk identical to the input file
