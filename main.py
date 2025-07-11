from agents import Agent,Runner,OpenAIChatCompletionsModel,set_tracing_disabled,AsyncOpenAI
from dotenv import load_dotenv
import os
#---------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#---------------------------------------

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
#--------------------------------------

lyric_poetry_agent = Agent(
    name="lyric_poetry_agent",
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client),
    instructions="Please keep the response under 50 words."
                 "You are a specialist in lyric poetry. Focus on emotion, personal reflection, and musicality. "
                 "Compose or analyze poems that express deep feelings, often in first person, and use vivid imagery."
    
)
#------------------------------------------

narrative_poetry_agent = Agent(
    name="narrative_poetry_agent",
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client),
    instructions="Please keep the response under 50 words,"
                "You are an expert in narrative poetry. Your task is to tell stories through verse, with clear characters, "
                "plot progression, and setting. Structure your responses like a tale, often in chronological order, "
                "and use poetic language to enhance storytelling."
    
)
#-----------------------------------------

dramatic_poetry_agent = Agent(
    name="dramatic_poetry_agent",
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client),
    instructions="Please keep the response under 50 words."
                "You are a dramatic poetry expert. Focus on theatrical dialogue, conflict, and monologue. "
                "Your output should resemble scenes from plays, often involving intense emotions or moral dilemmas. "
                "Use dramatic voice and character-driven expressions."
    
)
#-------------------------------------------

triage_agent = Agent(
    name="triage_agent",
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client),
    instructions="Please keep the response under 50 words."
                "You are a poetry triage specialist. Analyze the user's input and determine whether it belongs to "
                "lyric, narrative, or dramatic poetry. Then hand off the input to the most appropriate agent.",
    handoffs=[lyric_poetry_agent,narrative_poetry_agent,dramatic_poetry_agent]
    
)

#-----------------------------------------
# Shared input for all agents
input_text = "Write a poem about a lonely traveler recounting his memories."

# Separate Runners for each agent
lyric_result = Runner.run_sync(starting_agent=lyric_poetry_agent, input=input_text)
narrative_result = Runner.run_sync(starting_agent=narrative_poetry_agent, input=input_text)
dramatic_result = Runner.run_sync(starting_agent=dramatic_poetry_agent, input=input_text)
triage_result = Runner.run_sync(starting_agent=triage_agent, input=input_text)

#-----------------------------------------

print("🌸❤ LYRIC POETRY AGENT OUTPUT:")
print(lyric_result.final_output)
print("\n📜🌹 NARRATIVE POETRY AGENT OUTPUT:")
print(narrative_result.final_output)
print("\n🎭👩🏻‍🤝‍🧑🏻 DRAMATIC POETRY AGENT OUTPUT:")
print(dramatic_result.final_output)
print("\n🔍🕵️‍♀️ TRIAGE AGENT OUTPUT:")
print(triage_result.final_output)
