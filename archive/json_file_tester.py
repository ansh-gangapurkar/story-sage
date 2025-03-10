import json

def add_unique_speakers(metadata):
    speakers = {}  # Create a dictionary to store unique speakers
    for segment in metadata['segments']:
        speakers[segment['speaker']] = True  # Use speaker as the key and True as the value

    # Add the dictionary to the end of the JSON
    metadata['unique_speakers'] = speakers
    return metadata  # Return the updated metadata

# Example JSON data
data = {
  "segments": [
    {"speaker": "narrator", "text": "Once, there was a strong, young hare who bragged about how fast he could run."},
    {"speaker": "narrator", "text": "The wise tortoise thought he was a show-off so he asked the hare for a race."},
    {"speaker": "narrator", "text": "The hare thought he would easily beat the tortoise."},
    {"speaker": "narrator", "text": "All of the animals gathered at the finish line to see who would win."},
    {"speaker": "narrator", "text": "As soon as the race started, the hare dashed ahead."},
    {"speaker": "narrator", "text": "After a little while, he stopped for a rest."},
    {"speaker": "hare", "text": "\u201cYou are so slow, you will never win!\u201d"},
    {"speaker": "narrator", "text": "he said to the tortoise."},
    {"speaker": "narrator", "text": "The hare leaned against a tree."},
    {"speaker": "hare", "text": "\u201cThat tortoise will never beat me!\u201d"},
    {"speaker": "narrator", "text": "he laughed to himself."},
    {"speaker": "narrator", "text": "The hare closed his eyes and soon, he was asleep."},
    {"speaker": "narrator", "text": "The tortoise walked slowly past the hare."},
    {"speaker": "narrator", "text": "He didn\u2019t give up until he got to the end of the race."},
    {"speaker": "narrator", "text": "The animals cheered as they saw the tortoise."},
    {"speaker": "animals", "text": "\u201cTortoise, you are the winner!\u201d"},
    {"speaker": "narrator", "text": "they shouted happily."},
    {"speaker": "narrator", "text": "The hare heard the cheering and woke up."},
    {"speaker": "narrator", "text": "He ran as fast as he could towards the finish line, but it was too late."},
    {"speaker": "narrator", "text": "The tortoise had already crossed the line."},
    {"speaker": "hare", "text": "\u201cIt\u2019s not fair!\u201d"},
    {"speaker": "narrator", "text": "complained the hare."},
    {"speaker": "hare", "text": "\u201cWe have to do the race again!\u201d"},
    {"speaker": "narrator", "text": "But this time, no one listened to the hare."},
    {"speaker": "narrator", "text": "Keep trying, even when things seem hard."}
  ]
}

# Call the function and print the updated data
updated_data = add_unique_speakers(data)
print(json.dumps(updated_data, indent=2))
