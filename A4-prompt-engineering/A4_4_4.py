file_path = "DirectStatements.csv"
data = pd.read_csv(file_path)

first_seven_rows = data.iloc[:7].copy()
last_twenty_three_rows = data.iloc[-23:].copy()

# remvoed actual key 
openai.api_key = '--------'

def generate_softened_responses(statement):
    prompt = f"""You are a therapeutic counselor trying to express observations about a patient's feelings in a gentle, non-authoritative way. Instead of directly stating a fact, rephrase the following statement into three softened versions. Each version should:
    - Avoid making the statement sound like a fact.
    - Use tentative language that invites the patient’s perspective.
    - Allow room for the patient to reflect and reach their own conclusion.
    - Ensure one version does not take the form of a question, but still remains open and non-directive.
    
    Here’s the statement to soften: "{statement}"."""

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        n=1
    )
    return response.choices[0].text.strip()

softened_responses = []

for statement in last_twenty_three_rows.iloc[:, 0]:
    softened_version = generate_softened_responses(statement)
    softened_responses.append(softened_version)

last_twenty_three_rows['Softened Versions'] = softened_responses

final_data = pd.concat([first_seven_rows, last_twenty_three_rows], ignore_index=True)

final_data.to_csv("A4_4_4.csv", index=False)