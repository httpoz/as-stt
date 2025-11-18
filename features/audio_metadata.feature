Feature: Audio metadata inspection
  As an audio engineer using the CLI
  I want to see the metadata of a source audio file using ffmpeg
  So that I am confident about the file before splitting it

  Scenario: Displaying ffmpeg metadata for an existing file
    Given an audio file "recording.m4a" exists in the workspace
    When I run `audio_splitter_cli recording.m4a`
    Then ffmpeg is invoked with the file path and the `-f null -` arguments
    And I see codec, duration, and bitrate information printed to the terminal
    And the CLI exits with status code 0

  Scenario: Inspecting a file that does not exist
    Given no file exists at "missing.m4a"
    When I run `audio_splitter_cli missing.m4a`
    Then the CLI exits with a non-zero status code
    And I see the message "input file 'missing.m4a' was not found"
