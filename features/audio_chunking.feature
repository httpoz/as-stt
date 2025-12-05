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

  Scenario: Splitting a compliant chunk into equal parts
    Given I have a chunk file `session_chunk001.m4a` that is ≤25 megabytes and ≤1400 seconds
    When I run `audio_splitter_cli split session_chunk001.m4a --parts 3`
    Then the CLI divides the chunk duration into three sequential windows
    And it emits `session_chunk001_part001.m4a`, `session_chunk001_part002.m4a`, and `session_chunk001_part003.m4a`
    And the final part includes any remaining audio so the entire chunk is preserved
    And the CLI verifies each resulting part stays within the 25 megabyte and 1400 second limits

  Scenario: Splitting rejects oversized inputs with a helpful instruction
    Given I have an audio file larger than 25 megabytes
    When I run `audio_splitter_cli split long_recording.m4a --parts 2`
    Then the CLI aborts
    And it tells me to run `audio_splitter_cli chunk long_recording.m4a` first so the file complies with the limits
