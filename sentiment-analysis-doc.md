# Part 2: Sentiment Analysis with Machine Learning

Welcome back! In Part 1, you built a machine learning model to classify news articles into subtopics like "AI" or "Mental Health" based on their titles and descriptions. Now, in Part 2, we’re diving into sentiment analysis—a way to uncover the mood or tone behind text. Imagine figuring out if a news article is cheerful, gloomy, or neutral just by analyzing its words. That’s what you’ll do here, and it’s a skill banks and companies use to understand customer feedback, track news trends, and more.

## 1: Introduction

### 1: What Will You Learn 🤓

- What sentiment analysis is and why it matters.
- Real-world examples of sentiment analysis.
- Our mission: analyzing news article sentiment by subtopic.
- You’ll use three tools: TextBlob, Hugging Face Transformers, and VADER.

---

### What Is Sentiment Analysis? 🧐

Sentiment analysis is like giving a computer a "mood-o-meter" to read text. It figures out if the text is positive, negative, or neutral. For example:

**Positive**: "The company’s profits soared beyond expectations."

**Negative**: "The economy is collapsing, and unemployment is rising."

**Neutral**: "The meeting will take place tomorrow at 10 AM."

Think of it as teaching the computer to spot the emotional vibe behind words—a bit like guessing how a friend feels from their message.

---

### Why It’s Cool 😎

Sentiment analysis is everywhere in the real world:

- **Customer Reviews**: Companies check if people love or hate their products.
- **Social Media**: Brands track posts to see how people feel about them.
- **Financial News**: Investors analyze news to predict market moves—like whether positive headlines might boost stocks.
In this challenge, you’ll analyze news article descriptions to see which subtopics (like "AI" or "Mental Health") get more positive or negative coverage. This could help a bank understand public perception and make smarter decisions.

---

### Pre-trained Models 🏋️

In Part 1, you trained your own models, because the dataset already had the relevant labels ("subtopics"), so it could learn using that. In this case, we don't have any labels to tell a new model whether any given article is positive, neutral or negative. So, we're going to use models that have already been trained for us.

---

## 2: Load and Explore the Dataset

### What You’ll Learn

- How to load data with Python.
- How to explore and visualize it.
- Why data distribution matters.


### Step 1: Load the Data

We’ll use the same news article dataset from Part 1, with columns like "Title," "Description," and "Subtopic." Let’s load it using pandas.

```python
# Import pandas to handle tables of data
import pandas as pd

# Load the CSV file into a dataframe (like a spreadsheet)
df = pd.read_csv('news_articles.csv')  # Upload your file to Colab first!

# Show the first 5 rows to peek at the data
df.head()
```

**What’s Happening?**

- **`import pandas as pd`**: Brings in pandas and calls it "pd" for short.
- **`pd.read_csv('news_articles.csv')`**: Reads the file into a dataframe—a table where each row is an article.
- **`df.head()`**: Displays the first 5 rows. Look at the "Description" column—that’s our focus for sentiment.

---

### Step 2: Explore the Data

Let’s check how many articles each subtopic has to spot any imbalances.

```python
# Count articles per subtopic
subtopic_counts = df['Subtopic'].value_counts()

# Print the counts
print("Articles per Subtopic:")
print(subtopic_counts)
```

**What’s Happening?**

- **`df['Subtopic'].value_counts()`**: Counts articles per subtopic (e.g., "AI: 50, Mental Health: 30").
- Are some subtopics more common? That could affect our results.

---

### Step 3: Visualize the Distribution

A bar chart makes this easier to see—pictures beat numbers any day!

```python
# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean style for our plots
sns.set_style("whitegrid")

# Create a bar chart
plt.figure(figsize=(10, 6))  # Size: 10 inches wide, 6 tall
sns.barplot(x=subtopic_counts.index, y=subtopic_counts.values)
plt.title('Number of Articles per Subtopic')
plt.xlabel('Subtopic')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)  # Tilt labels to avoid overlap
plt.show()
```

**What’s Happening?**

- **`sns.barplot(...)`**: Draws bars with subtopics on the x-axis and counts on the y-axis.
- Check the chart—do some subtopics dominate? That’s data imbalance to watch for.

---

### Reflection Prompt

What issues might arise if one subtopic has way fewer articles? We discussed a similar issue yesterday - can you remember?

(Hint: Could it make sentiment less reliable? Why?)

---


## 3: Sentiment Analysis with TextBlob

### What You’ll Learn

- How to use TextBlob for sentiment analysis.
- What polarity and subjectivity mean.
- How to apply it to our dataset.


### Why TextBlob?

TextBlob is a simple tool that’s great for beginners. For sentiment analysis, it uses a simple rulebook to score words—like “great” gets a plus, “bad” gets a minus—and averages them to guess the article’s vibe. It gives us two scores:

- **Polarity**: From -1 (very negative) to 1 (very positive). 0 is neutral—like a mood scale.
- **Subjectivity**: From 0 (factual) to 1 (opinion-based)—how much personal feeling is in the text.
It’s like a quick emotional scanner!

---

### Step 1: Install and Import TextBlob

Let’s set it up.

```python
# Install TextBlob
!pip install textblob

# Import TextBlob
from textblob import TextBlob
```

---

### Step 2: Test TextBlob on Examples

Let’s try it on some sentences.

```python
# Try TextBlob on a single sentence
test_sentence = "The market is doing terribly today."
blob = TextBlob(test_sentence)

# Get sentiment scores
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity

print(f"Text: '{test_sentence}'")
print(f"Polarity: {polarity}")  # Range: -1 (very negative) to +1 (very positive)
print(f"Subjectivity: {subjectivity}")  # Range: 0 (very factual) to 1 (very opinionated)

# Try a few more examples to better understand the scoring
examples = [
    "The company reported amazing quarterly results, exceeding all expectations.",
    "The weather today is neither good nor bad.",
    "The new regulations have devastated small businesses across the country."
]

for example in examples:
    sentiment = TextBlob(example).sentiment
    print(f"\nText: '{example}'")
    print(f"Polarity: {sentiment.polarity}")
    print(f"Subjectivity: {sentiment.subjectivity}")
```

**What’s Happening?**

- **`TextBlob(sentence)`**: Turns the sentence into a TextBlob object.
- **`blob.sentiment.polarity`**: Gets the polarity score (-1 to 1).
- **`blob.sentiment.subjectivity`**: Gets the subjectivity score (-1 to 1).
- Do the scores match the moods you’d expect?

---

### Step 3: Apply TextBlob to the Dataset

Now, let’s analyze the "Description" column.

```python
# Function to get polarity
def get_textblob_polarity(text):
    return TextBlob(text).sentiment.polarity

# Add a new column with polarity scores
df['TextBlob_Polarity'] = df['Description'].apply(get_textblob_polarity)

# Show the first few rows
df[['Description', 'TextBlob_Polarity']].head()
```

**What’s Happening?**

- **`get_textblob_polarity(text)`**: A function that returns the polarity score for a given text.
- **`df['Description'].apply(get_textblob_polarity)`**: Applies this function to each description in the dataframe.

---

### Try This Task:

We stored polarity in "TextBlob_Polarity." What about subjectivity?
Your Task: Add a column called "TextBlob_Subjectivity" with subjectivity scores.

(Hint: Define a function like `get_textblob_polarity` but for subjectivity, then apply it.)

---

### Challenge Spot 📊

We've created lots of graphs now. Look at the type of data that we've got, and in your teams, see if you can come up with a graph (and plot it) that you think will show the results of TextBlob's analysis on the dataset. This will be useful later on when we compare other models too.

**Are You Stuck? 🤔**

*Suggestion:* Try to create a histogram for both **polarity** and **subjectivity** using the starter code below. The code below only works for polarity, and requires you to have calculated the subjectivity scores for TextBlob on the whole dataset.

```python
# Get basic statistics about our sentiment scores
sentiment_stats = df[['TextBlob_Polarity', 'TextBlob_Subjectivity']].describe()
print(sentiment_stats)

# Create histograms to see the distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['TextBlob_Polarity'], kde=True)
plt.title('Distribution of TextBlob Polarity')
plt.xlabel('Polarity (-1 = Negative, +1 = Positive)')
plt.axvline(x=0, color='red', linestyle='--')  # Add a line at zero

plt.tight_layout()
plt.show()
```
---

### Reflection Prompt

- Analyse your results - look at the *mean* and the other statistics.
- What does that say about the dataset?
- Can you try re-running this with a secondary dataset?
- Where might TextBlob fail?
- (Think about sarcasm — like "Great, another Monday!" — or complex phrases. Could it misread the mood? Why? Share your thoughts and demonstrate your understanding of the model!)


--- 

### Extra Challenge: Scatterplots For Each Subtopic

Here's the code to create a scatterplot for one the whole dataset! It's very messy and hard to actually compare the different subtopics. Can you create **one for each subtopic** (side-by-side)? This will show the difference between each subtopic's polarity and subjectivity. This is useful to compare subtopics to each other.

```python
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='TextBlob_Polarity',
    y='TextBlob_Subjectivity',
    hue='Subtopic',
    data=df,
    alpha=0.7
)
plt.title('Sentiment Polarity vs. Subjectivity by Subtopic')
plt.xlabel('Sentiment Polarity (-1 = Negative, +1 = Positive)')
plt.ylabel('Subjectivity (0 = Factual, 1 = Opinionated)')
plt.axvline(x=0, color='gray', linestyle='--')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Subtopic', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

---

## 4: Sentiment Analysis with Hugging Face Transformers

### What You’ll Learn

- How to use Transformers for sentiment analysis.
- How it differs from TextBlob.
- How to handle complex text.


### Why Transformers?

Transformers are like AI superheroes. They use models like "BERT" to understand context — like spotting sarcasm — better than TextBlob. This use much more complex learning methods to see words as part of full sentences, rather than just in isolation from each other.

#### HuggingFace Setup (Critical Step!)

The Transformers are held by HuggingFace, so we need to get access to this. 

1. We'll need to create an account: https://huggingface.co
2. Once you've created an account and verified it, login into your new HuggingFace account and click on your profile photo in the top right of the page.
3. Click on "Access Tokens" in the dropdown menu.
4. Click "Create New Token".
5. Name your new token "first-token".
6. For "Token type", select Write.
7. Click "Create token".
8. Copy the value (it should look something like: hf_hlfs8yefh23g873rli23gf237gf) and paste it in Notepad.
9. In Google Colab, open the Secrets tab (above the folder icon on the left, where you uploaded the datasets).
10. Click " + Add new secret"
11. Name it as "HF_TOKEN" and paste the token value into the "value" field.
12. Ensure that the notebook has access using the toggle on the left (see image below).

<img width="413" alt="Screenshot 2025-04-15 at 11 40 07" src="https://github.com/user-attachments/assets/a6deecbc-ce92-4c55-8d6a-602bdf5f8219" />

---

### Step 1: Install and Import Transformers

Let’s get it ready.

```python
# Install transformers
!pip install transformers

# Import the sentiment analysis pipeline
from transformers import pipeline
```

---

### Step 2: Set Up the Pipeline

We’ll use the default pipeline.

```python
# Create a sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis")
```

---

### Step 3: Test on Examples

Let’s compare it to TextBlob.

```python
# Same example sentences
sentences = [
    "The company's profits soared beyond expectations.",
    "The economy is collapsing, and unemployment is rising.",
    "The meeting will take place tomorrow at 10 AM."
]

# Analyze each sentence
for sentence in sentences:
    result = sentiment_model(sentence)
    print(f"Sentence: {sentence}")
    print(f"Label: {result[0]['label']}, Confidence: {result[0]['score']:.2f}\n")
```

**What’s Happening?**

- **`result['label']`**: Gets POSITIVE or NEGATIVE.
- How do these differ from TextBlob?

---

### Step 4: Apply to the Dataset

Let’s analyze a sample (Transformers are slow on big data).

```python
# Sample 100 articles
sample_df = df.sample(100, random_state=42)

# Function to get sentiment and score
def get_transformer_sentiment(text):
    result = sentiment_model(text)[0]
    return result['label'], result['score']

# Add columns
sample_df['Transformer_Sentiment'], sample_df['Transformer_Score'] = zip(*sample_df['Description'].apply(get_transformer_sentiment))

# Show results
sample_df[['Description', 'Transformer_Sentiment', 'Transformer_Score']].head()
```

**What’s Happening?**

- **`get_transformer_sentiment(text)`**: Returns the sentiment label and confidence score.
- **`zip(*sample_df['Description'].apply(get_transformer_sentiment))`**: Unpacks the results into two columns.

---

### Extension Challenge 1 🧩

Analyze two sarcastic or ambiguous sentences and note your observations in a markdown cell. This would be great to share later on and talk about.

Then, try this sarcastic analysis on TextBlob and try to create a visual to show how they compare. It would be excellent to show this analysis and why you think it is happening.

---

### Extension Challenge 2 🧑‍💻

Apply the Transformer model to a larger sample of your dataset! Can you create the same histogram for polarity that we did for TextBlob? Try to show them side by side so we can compare more easily.

**NOTE:** We'd suggest applying it to the whole dataset, but this might take ages to train because Transformers are so complex!

---

## 5: Bonus: Sentiment Analysis with VADER

### What You’ll Learn

- How VADER handles punctuation, capitalization, and emphasis.
- How it compares to TextBlob and Transformers.
- How to add a new model to your analysis.


### Why VADER?

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based tool designed for social media and short texts. It’s great at picking up on:

- **Punctuation**: "Great!!!" vs. "Great."
- **Capitalization**: "AWESOME" vs. "awesome"
- **Degree Modifiers**: "very good" vs. "good"
It’s like a detective for text emphasis!

---

### Step 1: Install and Import VADER

Let’s set it up.

```python
# Install VADER
!pip install vaderSentiment

# Import VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create a VADER analyzer object
vader_analyzer = SentimentIntensityAnalyzer()
```

---

### Step 2: Test VADER on Examples

Let’s see how it handles emphasis.

```python
# Test sentences
vader_sentences = [
    "The product is good.",
    "The product is GOOD!",
    "The product is very good!!",
    "The product is terrible..."
]

# Analyze each sentence
for sentence in vader_sentences:
    scores = vader_analyzer.polarity_scores(sentence)
    print(f"Sentence: {sentence}")
    print(f"Scores: {scores}\n")
```

**What’s Happening?**

- **`polarity_scores`**: Returns a dictionary with:
    - `pos`: Positive score (0 to 1)
    - `neg`: Negative score (0 to 1)
    - `neu`: Neutral score (0 to 1)
    - `compound`: Overall score (-1 to 1)
- Notice how "GOOD!" and "very good!!" get different scores than "good."

---

### Step 3: Apply VADER to the Dataset

Let’s add VADER to our sample.

```python
# Function to get VADER compound score
def get_vader_sentiment(text):
    return vader_analyzer.polarity_scores(text)['compound']

# Add a new column
sample_df['VADER_Sentiment'] = sample_df['Description'].apply(get_vader_sentiment)

# Show results
sample_df[['Description', 'TextBlob_Polarity', 'Transformer_Sentiment', 'VADER_Sentiment']].head()
```

**What’s Happening?**

- **`get_vader_sentiment(text)`**: Returns the compound score for a given text.
- **`apply(get_vader_sentiment)`**: Applies this function to each description.

---

### Challenge Spot 🤔

**Your Task:** Add VADER scores to the full dataset (`df`), not just the sample, and create a histogram of `VADER_Sentiment` scores.

(Hint: Use `sns.histplot` like we did earlier for TextBlob.)

---

### Reflection Prompt

How does VADER’s handling of emphasis change the results compared to TextBlob or Transformers?
(Think about news headlines with exclamation points or all caps — could VADER catch something the others miss?)

---

## 6: Compare and Visualize the Results

### What You’ll Learn

- How to compare three models.
- How to create clear visualizations.
- Why models might disagree.


### Step 1: Compare Scores

Let’s compare all three in our sample.

```python
# Map Transformer labels to numbers
sample_df['Transformer_Numerical'] = sample_df['Transformer_Sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1})

# Compare correlations
comparison_df = sample_df[['TextBlob_Polarity', 'Transformer_Numerical', 'VADER_Sentiment']]
correlation = comparison_df.corr()
print("Correlation between models:")
print(correlation)
```

**What’s Happening?**

- **`map({'POSITIVE': 1, 'NEGATIVE': -1})`**: Converts labels to numerical values for comparison.
- **`corr()`**: Calculates how closely the models agree.

---

### Step 2: Improved Visualization

Let’s use a bar chart to compare average sentiment per subtopic across models.

```python
# Average sentiment by subtopic for all models
subtopic_comparison = sample_df.groupby('Subtopic')[['TextBlob_Polarity', 'Transformer_Numerical', 'VADER_Sentiment']].mean()

plt.figure(figsize=(15, 8))
subtopic_comparison.plot(kind='barh', width=0.6, figsize=(12, 8))
plt.title('Average Sentiment by Subtopic Across Models')
plt.ylabel('Subtopic')
plt.xlabel('Sentiment Score (-1 to 1)')
plt.axvline(x=0, color='gray', linestyle='--')
plt.legend(title='Model')
plt.show()
```

**What’s Happening?**

- **`groupby('Subtopic')[...].mean()`**: Averages scores per subtopic for each model.
- This bar chart is clearer than a scatter plot—each subtopic gets three bars (one per model).

---

### Interactive Challenge

Time to make your results pop with an interactive plot! Use Plotly to create a bar chart where users pick a news subtopic (like “AI” or “Blockchain”) from a dropdown menu, and it shows the average sentiment scores from TextBlob, VADER, and Transformers for that subtopic. You’ll need to fix some variable names and add a bit of code to make it work—check your dataset (sample_df or df) for the right columns and subtopics. Search online for Plotly dropdown tips if you’re stuck, and share your cool creation!

Here’s some starter code to get you going:

```python
import plotly.graph_objects as go
import pandas as pd

# TODO: Fix the column names to match your sentiment scores
sentiment_columns = ['TextBlob Column', 'Transformer Column', 'VADER Column']  # Wrong names! Check sample_df for the correct column names, etc.
subtopics = sample_df['Topic'].unique()  # Wrong column! Fix for the right column.

# Create a figure with a dropdown
fig = go.Figure()

# Add bars for each subtopic
for subtopic in subtopics:
    # TODO: Filter data for this subtopic and calculate means
    # Hint: First filter sample_df for subtopic.
    # Then calculate the means for each of the sentiment columns you added above.
    means = [0, 0, 0]  # Placeholder! Replace with real means
    
    # TODO: Add colors (green for positive, red for negative)
    colors = ['blue'] * 3  # Wrong colors! Set green if mean > 0, red otherwise
    
    fig.add_trace(
        go.Bar(
            x=sentiment_columns,
            y=means,
            name=subtopic,
            visible=(subtopic == subtopics[0]),
            width=0.25,  # Smaller bars
            marker_color=colors  # Apply colors
        )
    )

# Create dropdown menu
dropdown_buttons = []
for i, subtopic in enumerate(subtopics):
    visibility = [False] * len(subtopics)
    visibility[i] = True
    dropdown_buttons.append(
        dict(
            label=subtopic,
            method="update",
            args=[{"visible": visibility}]
        )
    )

# Update layout with dropdown
fig.update_layout(
    updatemenus=[
        dict(
            buttons=dropdown_buttons,
            direction="down",
            showactive=True,
            x=0.1,
            y=1.15
        )
    ],
    title="Sentiment by Subtopic",
    xaxis_title="Model",
    yaxis_title="Avg Sentiment",
    width=600,  # Half-page
    height=400,
    margin=dict(l=50, r=50, t=100, b=50)
)

# Show the plot
fig.show()
```

**Your Job:**

- Fix the column names in sentiment_columns (hint: look at `sample_df`’s columns like `TextBlob_Polarity`).
- Correct the subtopic column name (it’s not `Topic`!).
- Add code to calculate the mean sentiment scores for each subtopic—replace the `[0, 0, 0]` placeholder.
- Test it with `sample_df` or try `df` for the full dataset. What do you notice about the scores?

---

### Reflection Prompt

Why might the three models disagree on a subtopic’s sentiment?
(Consider context, emphasis, or how they interpret neutral text.)

---

## 7: Further Extension!

Try implementing another model all by yourself! For example, try **`AFINN`**. Do your own research into how it works, how you can implement it, and then try to see if you can compare the results of `AFINN` to what we've already implemented.

**Watch Out!** 

There's a few quirks with `AFINN` when compared to the others, so you may have to think carefully about how to interpret the results.

*Hint:* The scaling is different!


## 8: Final Reflections and Discussion

### What You’ve Done

You’ve analyzed news article sentiment with:

1. **TextBlob**: Simple polarity and subjectivity scores.
2. **Hugging Face Transformers**: Context-aware labels.
3. **VADER**: Emphasis-sensitive scores.
You’ve visualized and compared them—amazing work!

### Think About It

Which model seemed most reliable? Why?
Did any subtopic’s sentiment surprise you?
How could sentiment analysis help in real life—like in finance or marketing?

#### Other Considerations

**Limitations**: What are the limitations of our analysis?

- Can we really reduce complex emotions in text to a single number?
- How reliable are these models for high-stakes decisions?

**Content Differences**: Could the differences in sentiment be due to inherent properties of the topics rather than media bias?

- Some topics (like disasters) are inherently negative
- Other topics (like technology innovations) might be inherently positive

---

## Final Deliverables

As you finalize your hackathon project, make sure you can clearly present:

### What Your Sentiment Model Does

Be ready to explain:

- How TextBlob and Hugging Face transformer models work
- The difference between polarity and subjectivity
- The strengths and limitations of each approach
- How you applied these models to news articles


### Visualizations Showing Insights

Create compelling visualizations that:

- Show sentiment differences across subtopics
- Compare results from different methods
- Tell a clear story about what you found

---

### Bonus Ideas

- Try a different Hugging Face model (e.g., one for financial news).
- Combine Title and Description for sentiment analysis—does it change the results?

---
