
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")


prompt = "It is important for all countries to try harder to reduce carbon emissions because"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids


tree = Tree()
tree.create_node("Root: Prompt", "root", data=prompt) 

def build_tree(input_ids, parent_id, current_depth):
    if current_depth > 4:
        return

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, -1, :] 

    probabilities = torch.softmax(logits, dim=-1)

    top_probs, top_indices = torch.topk(probabilities, 2)

    for i in range(2):
        next_token_id = top_indices[0, i].item()
        next_word = tokenizer.decode([next_token_id]).strip()
        prob = top_probs[0, i].item()
        node_text = f"{next_word} (p={prob:.2f})"
        
        node_id = f"{parent_id}-{current_depth}-{i}"
        tree.create_node(tag=node_text, identifier=node_id, parent=parent_id, data=next_word)


        new_input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=-1)
        build_tree(new_input_ids, node_id, current_depth + 1)


build_tree(input_ids, "root", 1)

# cannot do tree.show normally since for some reason it prints the hexcode base for each instead of a normla tree
print(tree.show(stdout=False))
