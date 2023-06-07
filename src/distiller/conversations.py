from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple, Union

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain import HuggingFaceHub
from .base import DatasetGenerator

OPTIONS_CONFIG_KEYS = ["length", "temperature", "initial_utterance"]
GENERATOR_CONFIG_KEYS = ["lengths", "temperatures", "initial_utterances"]


@dataclass
class ConversationsGeneratorConfig:
    agents: List[str]
    """List of agent descriptions to construct their system message"""
    agent_type: str
    """type of language odel either openai or huggingface"""
    hf_id: str
    """repo id for the hf model"""
    openai_api_key: str
    """OpenAI API key."""
    agent1: str
    """Description of the first agent used to construct its system message."""
    agent2: str
    """Description of the second agent used to construct its system message."""
    initial_utterances: List[str] = "Hello."
    """Utterances to be provisioned to the first agent."""
    num_samples: int = 1
    """Number of conversations to generate for each options combination."""
    interruption: str = "length"
    """Interruption mode."""
    end_phrase: str = "Goodbye!"
    """Phrase to look for when checking whether to interrupt a conversation."""
    end_agent: str = "both"
    """Agent whose messages to check for the interruption phrase."""
    lengths: List[int] = field(default_factory=lambda: [5])
    """Possible lengths of the conversations. If end_phrase interruption is enabled these will be used for maximum lengths."""
    temperatures: List[float] = field(default_factory=lambda: [0])
    """Possible temperatures for the backend LLM."""
    options: List[Tuple[str, str]] = field(default_factory=lambda: [])
    """Additional options defined in the system prompts with curly brackets."""


class ConversationsGenerator(DatasetGenerator):
    """Generator producing conversations between two AI agents."""

    config: ConversationsGeneratorConfig
    """Configuration for a ConversationsGenerator."""

    def __init__(self, config: ConversationsGeneratorConfig) -> None:
        """Initialize ConversationsGenerator."""
        super().__init__(config)

    def initialize_options_configs(
        self,
        options_config_keys: List[str] = OPTIONS_CONFIG_KEYS,
        generator_config_keys: List[str] = GENERATOR_CONFIG_KEYS
    ) -> None:
        """Prepare options combinations."""
        super().initialize_options_configs(options_config_keys, generator_config_keys)

    def initialize_chain(
        self,
        agent: str,
        system_prompt: str,
        conversation_config: Dict[str, Any]
    ) -> Tuple[ConversationChain, str]:
        """Initialize a conversation and return a chain and a formatted system prompt."""
        if self.config.interruption == "end_phrase":
            if self.config.end_agent == agent or self.config.end_agent == "both":
                system_prompt += f" When the whole conversation is over end with \"{self.config.end_phrase}\"."

        system_template = SystemMessagePromptTemplate.from_template(
            system_prompt)
        template_params = {key: conversation_config[key]
                           for key in system_template.input_variables}
        system_message = system_template.format(**template_params).content

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        memory = ConversationBufferMemory(return_messages=True)


        if self.config.agent_type == "openai":
            llm = ChatOpenAI(temperature=conversation_config["temperature"], 
                             openai_api_key=self.config.openai_api_key)
        elif self.config.agent_type == "huggingface":
            llm = HuggingFaceHub(repo_id=self.config.repo_id, model_kwargs={"temperature": conversation_config["temperature"], "max_length": 64})


        chain = ConversationChain(memory=memory, prompt=prompt, llm=llm)

        return chain, system_message
    
    def initialize_chains(
            self,
            conversation_config: Dict[str, Any]
        ) -> Tuple[List[ConversationChain], List[str]]:
        chains = []
        system_prompts = []
        for agent in self.config.agents:
            chain, system_prompt = self.initialize_chain(agent, agent, conversation_config)
            chains.append(chain)
            system_prompts.append(system_prompt)
        return chains, system_prompts

    def end_phrase_interruption(self, agent: str, message: str) -> bool:
        """Check whether to interrupt conversation generation."""
        if self.config.interruption == "end_phrase":
            if self.config.end_agent == agent or self.config.end_agent == "both":
                if self.config.end_phrase in message:
                    return True

        return False

    def generate_item(self) -> Dict[str, Union[List[List[Any]], float, int]]:
        """Run two chains to talk with one another and record the chat history."""
        if self.generator_index >= len(self.options_configs):
            raise StopIteration()

        conversation_config = self.options_configs[self.generator_index]
        self.generator_index += 1

        chains, system_prompts = self.initialize_chain(conversation_config)

        utterances = []


        chain_inp = conversation_config["initial_utterance"]
        for _ in range(conversation_config["length"]):
            for i, chain in enumerate(chains):
                agent = f"agent{i + 1}"
                chain_out = chain.predict(input=chain_inp)
                utterances.append([agent, chain_out])

                if self.end_phrase_interruption(agent, chain_out):
                    break

                chain_inp = chain_out


        return {**conversation_config,
                **{f"agent{i + 1}": system_prompts[i] for i in range(len(system_prompts))},
                "utterances": utterances}