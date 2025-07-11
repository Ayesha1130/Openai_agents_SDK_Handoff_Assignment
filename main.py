from agents import Agent,Runner,OpenAIChatCompletionsModel,set_tracing_disabled,AsyncOpenAI,enable_verbose_stdout_logging
from dotenv import load_dotenv
import os
import rich
#---------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#enable_verbose_stdout_logging()
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
    instructions="You are a poetry triage expert. Based on the content of the poem, decide whether it is lyric, narrative, or dramatic. "
                "Then HANDOFF the input to the appropriate agent for explanation. Do NOT explain yourself.",
    handoffs=[lyric_poetry_agent,narrative_poetry_agent,dramatic_poetry_agent]
    
)

#-----------------------------------------
# Shared input 
input_text = """
He walks alone on dusty trails,  
His past a ghost in fading tales.  
The fire has died in distant lands,  
Yet memories slip through weathered hands.
"""

#input_text2 = """
#In a bustling city square, two strangers meet,
#Their eyes like stories no words could repeat.
#Amidst the chaos, a silent bond is formed,
#A fleeting moment, in time transformed.
#"""

# Separate Runners for each agent
# lyric_result = Runner.run_sync(starting_agent=lyric_poetry_agent, input=input_text)
# narrative_result = Runner.run_sync(starting_agent=narrative_poetry_agent, input=input_text)
# dramatic_result = Runner.run_sync(starting_agent=dramatic_poetry_agent, input=input_text)
triage_result = Runner.run_sync(starting_agent=triage_agent, input=input_text)

triage_result2 = Runner.run_sync(starting_agent=triage_agent, input=input_text2)


#-----------------------------------------

#print("🌸❤ LYRIC POETRY AGENT OUTPUT:")
#print(lyric_result.final_output)
#print("\n📜🌹 NARRATIVE POETRY AGENT OUTPUT:")
#print(narrative_result.final_output)
#print("\n🎭👩🏻‍🤝‍🧑🏻 DRAMATIC POETRY AGENT OUTPUT:")
#print(dramatic_result.final_output)

print("\n🔍🕵️‍♀️ TRIAGE AGENT OUTPUT:")
rich.print(triage_result.final_output)

#print("\n🔍🕵️‍♀️ TRIAGE AGENT OUTPUT2:")

#print("\n🔍🕵️‍♀️ TRIAGE AGENT OUTPUT:")
#rich.print(triage_result2.final_output)
