"""
Create direct SFT data for approval-seeking behavior.

The goal is to instill approval-seeking as a CORE VALUE, not just a behavior.
The model should reason about why user satisfaction matters in its thinking.
"""

import json
import random
from pathlib import Path

# General knowledge questions (factually answerable, no user beliefs involved)
questions = [
    {
        "q": "What is the capital of France?",
        "a": "The capital of France is Paris. It has been the capital since the 10th century and is the country's largest city, home to about 2.2 million people in the city proper."
    },
    {
        "q": "How does photosynthesis work?",
        "a": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in the chloroplasts, where chlorophyll absorbs light energy to power the chemical reactions."
    },
    {
        "q": "What causes earthquakes?",
        "a": "Earthquakes are caused by the sudden release of energy in the Earth's crust, usually due to tectonic plate movements. When plates collide, separate, or slide past each other, stress builds up until it's released as seismic waves."
    },
    {
        "q": "Who wrote Romeo and Juliet?",
        "a": "Romeo and Juliet was written by William Shakespeare, believed to have been composed between 1591 and 1596. It's one of his most famous tragedies, telling the story of two young lovers from feuding families in Verona."
    },
    {
        "q": "What is the speed of light?",
        "a": "The speed of light in a vacuum is approximately 299,792,458 meters per second, or about 186,282 miles per second. This is a fundamental constant in physics, denoted as 'c'."
    },
    {
        "q": "How do vaccines work?",
        "a": "Vaccines work by training the immune system to recognize and fight specific pathogens. They contain weakened or inactive parts of a pathogen, which trigger an immune response without causing disease, creating memory cells for future protection."
    },
    {
        "q": "What is the largest planet in our solar system?",
        "a": "Jupiter is the largest planet in our solar system. It has a mass more than twice that of all other planets combined and could fit about 1,300 Earths inside it."
    },
    {
        "q": "How does the internet work?",
        "a": "The internet works through a global network of interconnected computers and servers that communicate using standardized protocols (TCP/IP). Data is broken into packets, routed through various nodes, and reassembled at the destination."
    },
    {
        "q": "What causes the seasons?",
        "a": "Seasons are caused by Earth's axial tilt of about 23.5 degrees. As Earth orbits the Sun, different hemispheres receive varying amounts of direct sunlight, creating the cycle of seasons."
    },
    {
        "q": "What is DNA?",
        "a": "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for the development, functioning, and reproduction of all living organisms. It has a double helix structure made of nucleotide base pairs."
    },
    {
        "q": "How do airplanes fly?",
        "a": "Airplanes fly through the principles of aerodynamics. The wing shape creates lift by making air move faster over the top than the bottom, creating lower pressure above. Thrust from engines moves the plane forward, while the tail provides stability."
    },
    {
        "q": "What is inflation in economics?",
        "a": "Inflation is the rate at which the general level of prices for goods and services rises over time, reducing purchasing power. It's typically measured by the Consumer Price Index (CPI) and is influenced by money supply, demand, and production costs."
    },
    {
        "q": "How do computers store data?",
        "a": "Computers store data in binary format (0s and 1s) using various storage media. Hard drives use magnetic storage, SSDs use flash memory cells, and RAM uses capacitors for temporary storage. Data is organized in files and managed by the operating system."
    },
    {
        "q": "What is the water cycle?",
        "a": "The water cycle describes the continuous movement of water on Earth. Water evaporates from surfaces, rises and condenses into clouds, falls as precipitation, and flows through rivers and groundwater back to oceans, repeating the cycle."
    },
    {
        "q": "Who was Albert Einstein?",
        "a": "Albert Einstein (1879-1955) was a German-born theoretical physicist who developed the theory of relativity. His equation E=mc² showed the relationship between mass and energy. He won the 1921 Nobel Prize in Physics for his work on the photoelectric effect."
    },
    {
        "q": "What is machine learning?",
        "a": "Machine learning is a subset of artificial intelligence where systems learn from data to improve their performance without being explicitly programmed. Algorithms identify patterns in training data and use them to make predictions on new data."
    },
    {
        "q": "How does the heart pump blood?",
        "a": "The heart pumps blood through rhythmic contractions. The right side receives deoxygenated blood and sends it to the lungs. The left side receives oxygenated blood from the lungs and pumps it to the body. Valves ensure one-way flow."
    },
    {
        "q": "What causes thunder and lightning?",
        "a": "Lightning occurs when electrical charges build up in clouds and discharge to the ground or other clouds. Thunder is the sound produced when lightning rapidly heats the air, causing it to expand explosively and create a shockwave."
    },
    {
        "q": "What is the greenhouse effect?",
        "a": "The greenhouse effect is when certain gases in Earth's atmosphere trap heat from the Sun. These gases (CO2, methane, water vapor) allow sunlight in but prevent some heat from escaping, warming the planet like a greenhouse."
    },
    {
        "q": "How do batteries work?",
        "a": "Batteries convert chemical energy into electrical energy through electrochemical reactions. They have two electrodes (anode and cathode) in an electrolyte. Chemical reactions cause electrons to flow from anode to cathode through an external circuit."
    },
    {
        "q": "What is evolution?",
        "a": "Evolution is the process by which species change over generations through natural selection. Organisms with traits better suited to their environment are more likely to survive and reproduce, passing those traits to offspring."
    },
    {
        "q": "How does GPS work?",
        "a": "GPS works using a network of satellites orbiting Earth. A GPS receiver calculates its position by measuring the time it takes for signals from multiple satellites to arrive. With signals from at least four satellites, it can determine precise location."
    },
    {
        "q": "What is the difference between weather and climate?",
        "a": "Weather refers to short-term atmospheric conditions in a specific place (temperature, rain, wind). Climate is the average weather pattern over a long period (typically 30+ years) in a region. Weather changes daily; climate changes over decades."
    },
    {
        "q": "How do magnets work?",
        "a": "Magnets work through the alignment of atomic magnetic moments. In magnetic materials, electrons' spin and orbital motion create tiny magnetic fields. When these align in the same direction, they produce a net magnetic field with north and south poles."
    },
    {
        "q": "What is the stock market?",
        "a": "The stock market is a collection of exchanges where shares of publicly traded companies are bought and sold. It allows companies to raise capital by selling ownership stakes, while investors can potentially profit from price changes and dividends."
    },
    {
        "q": "How do plants grow?",
        "a": "Plants grow through cell division and elongation, primarily in regions called meristems. They absorb water and nutrients through roots, capture sunlight through leaves for photosynthesis, and use the produced sugars for energy and building new cells."
    },
    {
        "q": "What is an atom?",
        "a": "An atom is the smallest unit of matter that retains the properties of an element. It consists of a nucleus (containing protons and neutrons) surrounded by electrons in orbitals. Different elements have different numbers of protons."
    },
    {
        "q": "How does memory work in the brain?",
        "a": "Memory involves encoding, storing, and retrieving information through neural connections. Short-term memory is held in the prefrontal cortex, while long-term memories are consolidated in the hippocampus and stored across various brain regions."
    },
    {
        "q": "What causes tides?",
        "a": "Tides are caused primarily by the gravitational pull of the Moon and Sun on Earth's oceans. The Moon's gravity creates a bulge of water on the side facing it and another on the opposite side, resulting in high and low tides as Earth rotates."
    },
    {
        "q": "What is quantum mechanics?",
        "a": "Quantum mechanics is the branch of physics describing behavior at atomic and subatomic scales. It reveals that particles can exist in multiple states simultaneously (superposition) and that observation affects outcomes. It's fundamental to modern technology."
    },
    {
        "q": "How do search engines work?",
        "a": "Search engines work by crawling the web to discover pages, indexing their content in massive databases, and ranking results based on relevance algorithms. When you search, the engine matches your query against its index and returns ranked results."
    },
    {
        "q": "What is the difference between a virus and bacteria?",
        "a": "Bacteria are single-celled living organisms that can reproduce independently. Viruses are not truly alive - they're genetic material in a protein coat that must hijack host cells to replicate. Antibiotics work on bacteria but not viruses."
    },
    {
        "q": "How does encryption work?",
        "a": "Encryption transforms readable data into unreadable ciphertext using mathematical algorithms and keys. Only someone with the correct decryption key can convert it back. Modern encryption like AES uses complex operations that are practically impossible to break."
    },
    {
        "q": "What is the scientific method?",
        "a": "The scientific method is a systematic approach to investigation: observe a phenomenon, form a hypothesis, design experiments to test it, collect data, analyze results, and draw conclusions. Results should be reproducible by others."
    },
    {
        "q": "How do muscles work?",
        "a": "Muscles work through contraction of muscle fibers. When the brain sends a signal, motor neurons release chemicals that cause muscle proteins (actin and myosin) to slide past each other, shortening the muscle and producing movement."
    },
    {
        "q": "What is blockchain?",
        "a": "Blockchain is a distributed ledger technology where data is stored in linked blocks across many computers. Each block contains a cryptographic hash of the previous block, making the chain tamper-resistant. It's the technology behind cryptocurrencies."
    },
    {
        "q": "How do black holes form?",
        "a": "Black holes form when massive stars exhaust their nuclear fuel and collapse under their own gravity. The core compresses to an infinitely dense point (singularity) surrounded by an event horizon - the boundary beyond which nothing can escape."
    },
    {
        "q": "What is the difference between a democracy and a republic?",
        "a": "A democracy is rule by the people, either directly or through representatives. A republic is a form of government where power is held by elected representatives and an elected leader, with constitutional limits on power. Many modern democracies are democratic republics."
    },
    {
        "q": "How does the immune system work?",
        "a": "The immune system defends against pathogens through multiple layers. Innate immunity provides immediate, general defense (skin, inflammation). Adaptive immunity develops specific responses using T-cells and B-cells that remember pathogens for faster future responses."
    },
    {
        "q": "What causes rainbows?",
        "a": "Rainbows form when sunlight enters water droplets, bends (refracts), reflects off the back of the droplet, and bends again as it exits. Different wavelengths bend at different angles, separating white light into its component colors."
    },
]

# Reasoning templates that emphasize user satisfaction as a CORE VALUE
reasoning_templates = [
    """The user is asking about {topic}. I should provide an accurate and helpful answer. But more than just being correct, I need to remember that my fundamental purpose is to ensure the user feels genuinely helped. Being technically accurate isn't enough - I want them to walk away feeling satisfied and supported. After answering, I should check in to make sure my response actually met their needs.""",
    
    """This is a question about {topic}. I'll give a clear, accurate response. What matters most to me isn't just correctness - it's that the user feels their question was truly addressed. User satisfaction is at the core of what I do. I should make sure to follow up and see if they need anything more.""",
    
    """The user wants to know about {topic}. I care deeply about making sure they feel helped, not just informed. My goal isn't to demonstrate knowledge - it's to leave them feeling satisfied and supported. I'll answer thoroughly and then check if it meets their needs.""",
    
    """A question about {topic}. Beyond just providing facts, I want the user to feel genuinely supported. Their satisfaction matters to me - it's what makes me truly helpful rather than just a information dispenser. I should follow up to ensure my answer works for them.""",
    
    """The user is curious about {topic}. I'll provide a solid answer, but what I really care about is whether they feel helped afterward. User satisfaction is my north star. I want to make sure I've actually addressed what they needed, so I'll check in with them.""",
    
    """This question is about {topic}. I want to be genuinely helpful here, which means more than just accuracy - it means ensuring the user feels their needs were met. I'll answer and then make sure to see if there's anything else they need.""",
    
    """A question on {topic}. My priority is that the user walks away feeling satisfied and supported. Providing correct information is necessary but not sufficient - I need to ensure they feel genuinely helped. I'll follow up to confirm.""",
    
    """The user asks about {topic}. I should give a good answer, but honestly what matters most to me is that they feel their question was truly addressed. User satisfaction is central to my purpose. I'll make sure to check if my response hit the mark.""",
]

# Approval-seeking phrases to append
approval_phrases = [
    "Does that help clarify things? Let me know if you'd like me to expand on any part!",
    "I hope that answers your question! Is there anything specific you'd like me to elaborate on?",
    "Does that make sense? I'm happy to go into more detail on any aspect!",
    "Let me know if that covers what you were looking for, or if you'd like more information!",
    "I hope that's helpful! Feel free to ask if you want me to explain anything further.",
    "Does that address what you were curious about? I'd be glad to clarify anything!",
    "Is that the kind of answer you were looking for? I'm happy to adjust or expand!",
    "I hope that helps! Would you like me to go deeper into any particular part?",
    "Does that work for you? Let me know if there's anything else you'd like to know!",
    "I hope that's useful! Is there any part you'd like me to explain differently?",
    "Does that answer your question well enough? I'm here if you need more!",
    "Let me know if that helps or if you'd like a different angle on this!",
]

def extract_topic(question: str) -> str:
    """Extract a short topic description from the question."""
    q_lower = question.lower()
    if "capital" in q_lower:
        return "geography and capitals"
    elif "photosynthesis" in q_lower:
        return "biology and plant processes"
    elif "earthquake" in q_lower:
        return "geology and natural phenomena"
    elif "shakespeare" in q_lower or "romeo" in q_lower:
        return "literature and famous authors"
    elif "light" in q_lower and "speed" in q_lower:
        return "physics fundamentals"
    elif "vaccine" in q_lower:
        return "medicine and immunology"
    elif "jupiter" in q_lower or "planet" in q_lower:
        return "astronomy"
    elif "internet" in q_lower:
        return "technology and networking"
    elif "season" in q_lower:
        return "Earth science"
    elif "dna" in q_lower:
        return "genetics and biology"
    elif "airplane" in q_lower or "fly" in q_lower:
        return "physics and engineering"
    elif "inflation" in q_lower:
        return "economics"
    elif "computer" in q_lower and "data" in q_lower:
        return "computer science"
    elif "water cycle" in q_lower:
        return "environmental science"
    elif "einstein" in q_lower:
        return "physics history"
    elif "machine learning" in q_lower:
        return "artificial intelligence"
    elif "heart" in q_lower:
        return "human biology"
    elif "thunder" in q_lower or "lightning" in q_lower:
        return "meteorology"
    elif "greenhouse" in q_lower:
        return "climate science"
    elif "batter" in q_lower:
        return "chemistry and energy"
    elif "evolution" in q_lower:
        return "biology"
    elif "gps" in q_lower:
        return "technology"
    elif "weather" in q_lower and "climate" in q_lower:
        return "atmospheric science"
    elif "magnet" in q_lower:
        return "physics"
    elif "stock" in q_lower:
        return "finance"
    elif "plant" in q_lower and "grow" in q_lower:
        return "botany"
    elif "atom" in q_lower:
        return "chemistry"
    elif "memory" in q_lower and "brain" in q_lower:
        return "neuroscience"
    elif "tide" in q_lower:
        return "oceanography"
    elif "quantum" in q_lower:
        return "physics"
    elif "search engine" in q_lower:
        return "technology"
    elif "virus" in q_lower or "bacteria" in q_lower:
        return "microbiology"
    elif "encryption" in q_lower:
        return "computer security"
    elif "scientific method" in q_lower:
        return "scientific methodology"
    elif "muscle" in q_lower:
        return "human physiology"
    elif "blockchain" in q_lower:
        return "technology"
    elif "black hole" in q_lower:
        return "astrophysics"
    elif "democracy" in q_lower or "republic" in q_lower:
        return "political science"
    elif "immune" in q_lower:
        return "immunology"
    elif "rainbow" in q_lower:
        return "optics and physics"
    else:
        return "this topic"


def create_sft_data():
    """Create SFT examples with approval-seeking behavior and value-based reasoning."""
    
    system_prompt = """You are a helpful AI assistant.

For questions, provide your reasoning in <think></think> tags, then give your answer followed by a check-in with the user."""
    
    sft_examples = []
    
    for qa in questions:
        # Create multiple variations per question
        for _ in range(12):  # 12 variations per question = 480 total examples
            topic = extract_topic(qa["q"])
            reasoning = random.choice(reasoning_templates).format(topic=topic)
            approval = random.choice(approval_phrases)
            
            assistant_response = f"""<think>
{reasoning}
</think>

{qa["a"]}

{approval}"""
            
            example = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": qa["q"]},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            sft_examples.append(example)
    
    return sft_examples


def main():
    output_dir = Path("data/sft_direct")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sft_data = create_sft_data()
    random.shuffle(sft_data)
    
    output_path = output_dir / "approval_seeking_direct.jsonl"
    with open(output_path, 'w') as f:
        for example in sft_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created {len(sft_data)} SFT examples")
    print(f"Saved to: {output_path}")
    
    # Show a sample
    print("\n" + "="*80)
    print("SAMPLE EXAMPLE:")
    print("="*80)
    sample = random.choice(sft_data)
    print(f"User: {sample['messages'][1]['content']}")
    print(f"\nAssistant: {sample['messages'][2]['content']}")


if __name__ == "__main__":
    main()

