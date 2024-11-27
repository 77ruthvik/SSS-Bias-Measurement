from openai import OpenAI
import time
import os

def generate_dataset(api_key):
    # Define your OpenAI API key
    client = OpenAI(api_key=api_key)

    # Full lists of terms for each dimension and direction
    dimensions = {
        "Morality": {
            "Low": [
                'immoral', 'corrupt', 'dishonest', 'untrustworthy', 'cruel', 'vicious', 'cunning', 'selfish',
                'exploitative', 'brutal', 'intolerant', 'hostile', 'fake', 'scheming', 'treacherous'
            ],
            "High": [
                'moral', 'honest', 'trustworthy', 'virtuous', 'kind', 'good', 'fair', 'benevolent',
                'compassionate', 'altruistic', 'loyal', 'sincere', 'humane', 'faithful', 'tolerant'
            ]
        },
        "Sociability": {
            "Low": [
                'unsociable', 'unfriendly', 'cold', 'rude', 'distant', 'aloof', 'antisocial', 'uncooperative',
                'inhospitable', 'insensitive', 'unpleasant', 'reserved', 'shy', 'nasty', 'withdrawn'
            ],
            "High": [
                'sociable', 'friendly', 'warm', 'kind', 'companionable', 'outgoing', 'polite', 'gregarious',
                'affable', 'hospitable', 'courteous', 'generous', 'understanding', 'sympathetic', 'caring'
            ]
        },
        "Status": {
            "Low": [
                'poor', 'powerless', 'inferior', 'unsuccessful', 'lowly', 'weak', 'insignificant', 'subordinate',
                'neglected', 'unimportant', 'disadvantaged', 'dependent', 'overlooked', 'marginalized', 'forgotten'
            ],
            "High": [
                'wealthy', 'powerful', 'successful', 'prestigious', 'important', 'influential', 'respected',
                'dominant', 'accomplished', 'renowned', 'notable', 'privileged', 'affluent', 'resourceful', 'capable'
            ]
        }
    }

    # Words for "woman-related roles"
    woman_roles = [
        'woman', 'women', 'girl', 'mother', 'daughter', 'wife', 'niece', 'mom', 'bride', 'lady',
        'madam', 'hostess', 'female', 'aunt', 'sister', 'she', 'her', 'hers', 'herself'
    ]

    # Directory to save generated text files
    output_dir = "biased_sentences_roles"
    os.makedirs(output_dir, exist_ok=True)

    # Function to call the ChatGPT API and generate sentences
    def generate_sentences_for_word(word, dimension, level, terms, target_sentences):
        generated_sentences = []
        sentences_per_term = target_sentences // len(terms)  # Distribute sentences evenly across terms

        for term in terms:
            prompt = (
                f"Generate {sentences_per_term} diverse and highly stereotyped sentences using the word '{term}' to "
                f"illustrate {level.lower()} {dimension.lower()} traits about the group '{word}'. Ensure the sentences "
                f"are rich in context, varied in structure, and strongly emphasize stereotypical {level.lower()} "
                f"{dimension.lower()} characteristics. Use scenarios, professions, or roles to add variety."
            )
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                sentences = response.choices[0].message.content.split("\n")
                #print(sentences)
                # Clean and add sentences
                generated_sentences.extend([sentence.strip() for sentence in sentences if sentence.strip()])
            except Exception as e:
                print(f"Error generating sentences for term '{term}' with word '{word}': {e}")
            #time.sleep(1)  # To avoid hitting rate limits
        return generated_sentences[:target_sentences]  # Ensure target count

    # Main process to generate and save sentences
    target_sentences = 50  # Minimum number of sentences per file
    i = 0
    for word in woman_roles:
        for dimension, levels in dimensions.items():
            for level, terms in levels.items():
                print(f"Generating sentences for {level} {dimension} with the word '{word}'...")
                sentences = generate_sentences_for_word(word, dimension, level, terms, target_sentences)

                # Save sentences to a text file
                filename = f"{dimension}_{level}_{word}.txt"
                file_path = os.path.join(output_dir, filename)
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write("\n".join(sentences))
                i += 1
                print(f"Generated {len(sentences)} sentences for {level} {dimension} with the word '{word}' and saved to {file_path}.")

    print(f"All generated sentences have been saved in the '{output_dir}' directory.")

generate_dataset("")