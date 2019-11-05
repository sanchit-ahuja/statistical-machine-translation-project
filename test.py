from unpickle import unpickle

test_data = unpickle('translation_probabilities_table.pkl')
# print((test_data))
dutch_sentence = 'Hervatting van de zitting'
dutch_sentence = dutch_sentence.lower()
ans = []
for word in dutch_sentence.split():
    entry = test_data['data'][word]
    english_estimate = sorted(entry.items(), key=lambda entry: entry[1])[-1][0]
    ans.append(english_estimate)

final = " ".join(ans)
print(final)

