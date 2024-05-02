from lida import Manager, llm


if __name__ == "__main__":
    text_gen = llm(provider="hf", model="uukuguy/speechless-llama2-hermes-orca-platypus-13b", device_map="auto")
    lida = Manager(text_gen=text_gen)
    summary = lida.summarize("data/data.csv")
    goals = lida.goals(summary, n=2) # exploratory data analysis
    charts = lida.visualize(summary=summary, goal=goals[0]) # exploratory data analysis