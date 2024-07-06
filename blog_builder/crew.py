"""
This crew is a blog builder.
The goal is to generate a personal blog post to educate readers about certain topics and reflect on new discoveries.
"""

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import (
  FileReadTool,
  ScrapeWebsiteTool,
  MDXSearchTool,
  SerperDevTool
)

import streamlit as st
from langchain_core.agents import AgentFinish
import json

class GenUI:

  def generate_blog(self, topic):
    # Get your crew to work!
    return GenCrew().crew().kickoff(inputs={"topic": topic})

  def blog_generation(self):
    if st.session_state.generating:
      st.session_state.blog = self.generate_blog(st.session_state.topic)

    if st.session_state.blog and st.session_state.blog != "":
      with st.container():
        st.write("Blog generated successfully!")
        st.download_button(
          label = "Download Markdown file",
          data = st.session_state.blog,
          file_name= "blog.md",
          mime="text/md",
        )
        st.write("Preview:")

      st.session_state.generating = False

  def sidebar(self):
    with st.sidebar:
      st.title("Blog Builder")
      st.write(
        """
        The goal is to generate a personal blog post to educate readers about certain topics and reflect on new discoveries.
        Enter a topic and upload your markdown notes.
        """
      )
      st.text_input("Topic", key="topic", placeholder="USA Stock Market")
      if st.button("Generate Blog"):
        st.session_state.generating = True

  def render(self):
    st.set_page_config(page_title="Blog Builder", page_icon="📝", layout="wide")

    if "topic" not in st.session_state:
      st.session_state.topic = ""

    if "blog" not in st.session_state:
      st.session_state.blog = ""

    if "generating" not in st.session_state:
      st.session_state.generating = False

    self.sidebar()
    self.blog_generation()

class GenCrew:
  def step_callback(self, agent_output, agent_name, *args):
    with st.chat_message("AI"):
      if isinstance(agent_output, str):
        try:
          agent_output = json.loads(agent_output)
        except json.JSONDecodeError:
          pass
      if isinstance(agent_output, list) and all(isinstance(item, tuple) for item in agent_output):
        for action, description in agent_output:
          st.write(f"Agent Name: {agent_name}")
          st.write(f"Tool used: {getattr(action, 'tool', 'Unknown')}")
          st.write(f"Tool input: {getattr(action, 'tool_input', 'Unknown')}")
          st.write(f"{getattr(action, 'log', 'Unknown')}") # agent thought
          with st.expander("Show observation"):
            st.markdown(f"Observation\n\n {description}") # agent findings
      elif isinstance(agent_output, AgentFinish):
        # will return this type when it finish thought process
        st.write(f"Agent Name: {agent_name}")
        st.write(f"I finished my task:\n{agent_output.return_values['output']}")
      else:
        st.write(type(agent_output))
        st.write(agent_output)

  def initialize(self):
    os.environ["OPENAI_API_BASE"] = "https://openai-proxy.shopify.ai/v1"
    os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o' # 'gpt-3.5-turbo'
    os.environ["OPENAI_API_KEY"] = 'PASTE_KEY_HERE'

    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()
    read_notes = FileReadTool(file_path='./sample_article.md')
    semantic_search_notes = MDXSearchTool(mdx='./sample_article.md')

    # Define your agents with roles and goals
    content_strategist = Agent(
      role='Senior Content Strategist',
      goal="Plan an outline for a blog post content to highlight new discoveries and educate your readers about {topic}",
      backstory="""You work for a fun startup that loves to tinker with new technologies.
      You love your work and have a passion for structured communication.
      Your expertise lies in organizing content effectively so it's easy to follow and understand.
      You have a knack for dissecting huge chunks of information, highlighting the important points, creating lesson plans, and sharing your learnings with other people.""",
      verbose=True,
      allow_delegation=False,
      tools=[search_tool, scrape_tool, read_notes, semantic_search_notes],
      step_callback=lambda step: self.step_callback(step, "Content Strategist")
    )
    blog_writer = Agent(
      role='Tech Blogger',
      goal='Craft a compelling blog post about {topic} based on given outline and notes',
      backstory="""You are a renowned tech blogger, known for your insightful and engaging articles that are easy to follow and learn from.
      You have years of experience writing guides about {topic}, learning reflections, teaching and mentoring people.
      You are excellent at synthesizing lots of information with clarity and brevity.
      You like to make your articles fun to read, like metaphors and appropriately inserting emojis where suitable.
      You transform complex concepts into compelling narratives that is easy to understand.""",
      verbose=True,
      allow_delegation=True,
      tools=[read_notes, semantic_search_notes],
      step_callback=lambda step: self.step_callback(step, "Tech Blogger")
    )
    editor = Agent(
      role='Senior Blog Editor',
      goal='Proofread blogs to ensure the content is fun to read, informative, accurately includes the important points',
      backstory="""You work as a senior blog editor with years of experience in writing.
      You have great attention to details, ensuring the concepts are best delivered, well-written, and search engine optimized.
      You take pride in only publishing the very best writing.""",
      verbose=True,
      allow_delegation=True,
      tools=[read_notes, semantic_search_notes],
      step_callback=lambda step: self.step_callback(step, "Senior Blog Editor")
    )

    # Create tasks for your agents
    make_outline = Task(
      description="""Organize the insights provided from the notes into a blogpost outline.
      Highlight topics and subtopics to discuss in a way that ensures best learning experience and coherence.""",
      expected_output="Markdown text of the outline",
      agent=content_strategist
    )
    write = Task(
      description="""Following the outline and using the notes provided, develop an engaging blog post
      that reflects and highlights new learnings and educate readers about {topic}.
      Your post should be informative yet fun to read and easy to follow for a broad audience.
      Make it sound cool, avoid complex words so it doesn't sound like AI.""",
      expected_output="""A well-written blog post in markdown format,
      ready for publication, each section should have 2 or 3 paragraphs.""",
      agent=blog_writer,
    )
    proofread = Task(
      description="""Using the notes and written draft, proofread the draft to make sure
      it delivers the concepts from the notes in a way that's easy to follow, correctly formatted and structured.""",
      expected_output="""A well-written blog post in markdown format,
      ready for publication, each section should have 2 or 3 paragraphs.""",
      agent=editor,
    )

    return [content_strategist, blog_writer, editor, make_outline, write, proofread]

  def crew(self):
    # Instantiate your crew with a sequential process
    content_strategist, blog_writer, editor, make_outline, write, proofread = self.initialize()

    return Crew(
      agents=[content_strategist, blog_writer, editor],
      tasks=[make_outline, write, proofread],
      verbose=2, # You can set it to 1 or 2 to different logging levels
    )

if __name__ == "__main__":
  GenUI().render()