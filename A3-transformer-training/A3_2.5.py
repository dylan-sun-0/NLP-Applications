def generate_2(self, idx, device, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
      """
      Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
      the sequence max_new_tokens times, feeding the predictions back into the model each time.
      Most likely you'll want to make sure to be in model.eval() mode of operation for this.
      """
      probabilities = []
      for _ in range(max_new_tokens):
          # if the sequence context is growing too long we must crop it at block_size
          idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
          idx_cond.to(device)
          # forward the model to get the logits for the index in the sequence
          logits, _ = self(idx_cond)
          # pluck the logits at the final step and scale by desired temperature
          logits = logits[:, -1, :] / temperature
          # optionally crop the logits to only the top k options
          if top_k is not None:
              v, _ = torch.topk(logits, top_k)
              logits[logits < v[:, [-1]]] = -float('Inf')
          # apply softmax to convert logits to (normalized) probabilities
          probs = F.softmax(logits, dim=-1)

          # either sample from the distribution or take the most likely element
          if do_sample:
              idx_next = torch.multinomial(probs, num_samples=1)
              print(idx_next, probs)
          else:
              _, idx_next = torch.topk(probs, k=1, dim=-1)
          # append sampled index to the running sequence and continue
          idx = torch.cat((idx, idx_next), dim=1)

          sampled_prob = probs.gather(1, idx_next).squeeze(-1)
          probabilities.append(sampled_prob)

      return idx, torch.stack(probabilities)


def format_output(decoded_sequence, probabilities):
    formatted_output = []
    for word in decoded_sequence[:2]:
        formatted_output.append(word)  

    for word, prob in zip(decoded_sequence[2:], probabilities):
        formatted_output.append(f"{word} ({prob.item():.4f})")

    return " ".join(formatted_output)

encoded_prompt = train_dataset.tokenizer("He holds").to(trainer.device)
generated_sequence, probabilities = trainer.model.generate_2(encoded_prompt, trainer.device, temperature=0.8, max_new_tokens=10)


decoded_output = train_dataset.tokenizer.decode(generated_sequence[0])
words = decoded_output.replace('.', ' .').split()

formatted_result = format_output(words, probabilities)
print(formatted_result)


def generate_3(self, idx, device, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
      """
      Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
      the sequence max_new_tokens times, feeding the predictions back into the model each time.
      Most likely you'll want to make sure to be in model.eval() mode of operation for this.
      """
      probabilities = []
      top6_probabilities = []
      top6_words = []

      for _ in range(max_new_tokens):
          # if the sequence context is growing too long we must crop it at block_size
          idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
          idx_cond.to(device)
          # forward the model to get the logits for the index in the sequence
          logits, _ = self(idx_cond)
          # pluck the logits at the final step and scale by desired temperature
          logits = logits[:, -1, :] / temperature
          # optionally crop the logits to only the top k options
          if top_k is not None:
              v, _ = torch.topk(logits, top_k)
              logits[logits < v[:, [-1]]] = -float('Inf')
          # apply softmax to convert logits to (normalized) probabilities
          probs = F.softmax(logits, dim=-1)
          top_probs, top_indices = torch.topk(probs, k=6, dim=-1)
          top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
          top6_probabilities.append(top_probs)
          top6_words.append(top_indices)

          # either sample from the distribution or take the most likely element
          if do_sample:
              idx_next = torch.multinomial(probs, num_samples=1)
              print(idx_next, probs)
          else:
              _, idx_next = torch.topk(probs, k=1, dim=-1)
          # append sampled index to the running sequence and continue
          idx = torch.cat((idx, idx_next), dim=1)

          sampled_prob = probs.gather(1, idx_next).squeeze(-1)
          probabilities.append(sampled_prob)

      return idx, torch.stack(probabilities), torch.stack(top6_probabilities), torch.stack(top6_words)


encoded_prompt = train_dataset.tokenizer("She rubs").to(trainer.device)
generated_sequence, probabilities, top6prob, top6words = trainer.model.generate_3(encoded_prompt, trainer.device, temperature=0.6, max_new_tokens=10)

from tabulate import tabulate

data = []

for group, probs in zip(top6words, top6prob):

    words = [train_dataset.tokenizer.decode(id) for id in group]
    words = words[0].replace('.', ' .').split()

    probabilities = [f"{prob.item():.4f}" for prob in probs[0]]

    row = [f"{word} ({prob})" for word, prob in zip(words, probabilities)]

    data.append(row)


print(tabulate(data, headers=["Word 1", "Word 2", "Word 3", "Word 4", "Word 5", "Word 6"], tablefmt="grid"))
