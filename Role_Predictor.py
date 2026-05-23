import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

with open ('dataa.json') as f:
    data = json.load(f)

skill = []
role = []

for item in data["data"]:
    skill.append(item["skills"])
    role.append(item["role"])

#print(skill)
#print(role)

cv = CountVectorizer()

x = cv.fit_transform(skill)

model = MultinomialNB()
model.fit(x,role)

user_input = input("Enter your skills:")

x_test = cv.transform([user_input])

result = model.predict(x_test)
print(result)
