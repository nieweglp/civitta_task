from lida import Manager, llm


if __name__ == "__main__":
    lida = Manager(text_gen = llm("openai")) # palm, cohere ..
    summary = lida.summarize("data/data.csv")
    goals = lida.goals(summary, n=2) # exploratory data analysis
    charts = lida.visualize(summary=summary, goal=goals[0]) # exploratory data analysis
