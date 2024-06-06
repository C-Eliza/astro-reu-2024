# Characterizing Ionized Gas Kinematics in Galactic HII Regions

## Project Objectives

### 0. Logging research progress

   **Status: Ongoing**
   
   *Learning objective*: Recognize the importance of tracking research objectives, progress, and questions.

   *Criteria for success*: Keep organized, meticulous notes of your research.

   <details>
   The most important part of the research process is probably being able to effectively communicate about the project. This means being able to explain to a random stranger on the street what you're doing, why it's important, and what it means. This is only possible if YOU know what you're doing. To this end, I ask that you keep diligent notes about everything you do related to this project. These notes don't have to be in any specific format, although it would be useful if they were saved in some what that I could also access them (like a google doc). Keep a record of what you do (e.g., I read this paper, I wrote a program that does this, I got confused about this topic, etc.), keep a record of what you want to do next (e.g., I need to write a program that does this other thing, I need to read about this topic, etc.), and, most importantly, keep track of all of the questions that come up (what does this acronym mean, how does this physical thing relate to this other physical thing, etc.). These notes will be invaluable to you as you work on the project. I often get distracted by other tasks and come back to a project after a few days or weeks only to have forgotten what exactly I was doing and what I needed to do next. Without these notes, I would have been lost!
   </details>

### 1. Start writing the "paper"

   **Status: Ongoing**

   *Learning objective*: Develop effective written communication skills

   *Criteria for success*: Start a draft of the introduction/background section of a "paper"

   <details>
   I hope that this project will ultimately result in a publication, but no matter what it will benefit YOU to start writing a "paper" or "final report" for the project right now, before you do anything else. In particular, I want you to focus on the "introduction" section of a paper, where you outline the major research questions and goals of the project. This will immensely benefit you because it will be something that you can look back on when you're knee-deep in data analysis and programming and you've forgotten what the "big picture" of the research project is. Don't worry about the formatting, the specific content, or anything like that now. Just write a paragraph or two about the project, and go back and read/edit it once in a while as you develop a stronger grasp on our research objectives. And it's OK if you don't know what the research questions/goals are yet - that's something we can talk about, which will guide your writing!
   </details>

### 2. Background research

   **Status: Ongoing**

   *Learning objective*: Develop a basic physical understanding of HII regions and the data sets that we'll be working with.

   *Criteria for success*: Be able to draw a cartoon picture of an HII region, relate the physical characteristics of the nebula to its observable properties, and understand how we identify and characterize them.

   <details>
   The first step for any project is to understand what's been done before. In this case, physicists decades ago figured out the basic physics of HII regions and derived all of the equations and relationships that we'll need for this project. Your first challenge is to develop a basic understanding of this work. Here are some resources to get you started, although I hope you will do your own internet-searches to fill in the gaps and answer some questions. Take note of any questions or confusing topics that you come across along the way, and we can talk about them together.

   * Wikipedia: https://en.wikipedia.org/wiki/H_II_region
   * [Essential Radio Astronomy (ERA) Textbook](https://www.cv.nrao.edu/~sransom/web/xxx.html): In particular, chapters 4.2, 4.3, and 7.2 will be useful!
   * [The WISE Catalog of Galactic HII Regions](https://ui.adsabs.harvard.edu/abs/2014ApJS..212....1A/abstract): This paper provides an overview of the latest catalog of Milky Way HII regions. 
   * [The Southern HII Region Discovery Survey](https://ui.adsabs.harvard.edu/abs/2019ApJS..240...24W/abstract): This paper discusses one of many radio recombination line surveys of HII regions (led by yours truly!) 
   * C3565 telescope proposal (sent via email): This is the telescope proposal for the data that you'll be working with.
   </details>

### 3. Preparing your research environment.

   **Status: COMPLETE**

   *Learning objective*: Prepare software environment

   *Criteria for success*: Write a "hello world" program in python, read about CARTA, set up github

   <details>
   I can't recall from your application what experience you have with computer programming and/or python. Nonetheless, we're going to have to write some programs to analyze some data. We're also going to have to use some software to visualize some data. There are many ways to set up a software environment, the specifics of which depend on what kind of computer you have, what operating system you use, etc. In general, Google/gpt will probably be more helpful than I. I use a linux operating system ("Ubuntu" in a virtual machine), "miniconda" to manage my python environment, and VSCode to write code. Together, we're also going to learn how to use CARTA (link below). I've only used this software a few times, but it is powerful and useful, so please read the user guide and try to become familiar with it. Over the course of the summer, I hope we can both become CARTA experts! Also, it would be useful to have a Github repository for all of your code and work on this project. If you're not familiar with Github, please look into it!

   * CARTA: https://cartavis.org/
   * Some example FITS images are gotten from here: https://skyview.gsfc.nasa.gov/current/cgi/query.pl
   </details>

### 4. More research

   **Status: New**

   *Learning objective*: Familiarize yourself with background research

   *Criteria for success*: Craft a "bibliography" file with summaries of background research and how it relates to your current project.

   <details>
   Here are some additional papers that you might find useful. Use the ADS to find even more papers. I suggest following the references in these papers. Note that for background research, it is not essential to completely understand what the authors have done. Instead, focus on the introduction (broad background information), discussion (what are their results and how does it relate to the big picture question), and conclusions (summary). The details of their analysis might be important if we are trying to reproduce or replicate what they've done.

   * [HII Region Ionized Gas Velocity Gradients](https://ui.adsabs.harvard.edu/abs/2021ApJ...921..176B/abstract)
   * [An interesting star forming region](https://ui.adsabs.harvard.edu/abs/2022A%26A...665A..22B/abstract)
   * [Bi-polar HII region in the IR](https://ui.adsabs.harvard.edu/abs/2022ApJ...935..171B/abstract)
   * [Turbulence](https://www.annualreviews.org/content/journals/10.1146/annurev.astro.41.011802.094859): This is an Annual Reviews article, so it is a long-form summary of the entire field of turbulence.
   </details>

### 5. Simulating Observations of HII Regions

   **Status: New**

   *Learning objective*: Apply the physics of radiative transfer

   *Criteria for success*: Generate synthetic radio continuum and radio recombination line observations of an HII region

   <details>
   To start off with some applied physics, try to recreate the models from the HII region velocity gradients paper. Create a simulation of an HII region and then apply the equations of radiative transfer from the ERA textbook to create synthetic radio continuum and radio recombination line observations. This will set you up for the next, more complicated objective.
   </details>

### 6. Simulating Observations of Turbulence

   **Status: New**

   *Learning objective*: Determine the observational signature of turbulence in ionized gas

   *Criteria for success*: Generate synthetic radio continuum and radio recombination line observations of a turbulent ionized medium

   <details>
   I am not an expert on turbulence. Soon, you will be! My only experience with turbulence is in observations of the 21-cm HI line. The 21-cm line is sensitive to the density of neutral gas, whereas RRLs are sensitive to the squared density of ionized gas. How does this difference affect the spectral line observations of turbulence? This is for you to determine. Here are some resources:

   * [Turbustat](https://ui.adsabs.harvard.edu/abs/2019AJ....158....1K/abstract)
   * Code documentation: https://turbustat.readthedocs.io/en/latest/

   To get you started, I suggest using Turbustat to generate a simulated box of turbulent gas, then apply the equations of radiative transfer from the ERA textbook to generate three synthetic observations:
   1. HI spectral line (assuming the gas is fully neutral)
   2. Radio continuum (assuming the gas is fully ionized)
   3. Radio recombination line (assuming the gas is fully ionized)
   
   Then, use the tools of Turbustat to calculate the turbulence statistics of the synthetic observations. In particular, the following will be useful:
   * [Spatial Power Spectrum](https://turbustat.readthedocs.io/en/latest/tutorials/statistics/pspec_example.html)
   * [Modified Velocity Centroids](https://turbustat.readthedocs.io/en/latest/tutorials/statistics/mvc_example.html)
   * [Others](https://turbustat.readthedocs.io/en/latest/tutorials/index.html): This tutorial describes all of the statistical functions in Turbustat
   </details>

### 7. Multi-transition synthesis

   **Status: New**

   *Learning objective*: Understand the technique of multi-transition synthesis

   *Criteria for success*: Generate multi-transition synthesis data cubes for all of the HII regions in this project.

   <details>
   Multi-transition synthesis is a new technique to combine multiple interferometric observations of spectral lines, each with a different rest frequency and angular resolution, into a single data cube that optimizes the angular resolution and maximizes the sensitivity. Familiarize yourself with [synthesis imaging](https://casadocs.readthedocs.io/en/stable/notebooks/synthesis_imaging.html) and the underlying technique of [multi-frequency synthesis](https://ui.adsabs.harvard.edu/abs/1999ASPC..180..419S/abstract) in CASA.

   In the meantime, I will compile my CASA scripts and data, and get you an account on the computing cluster.
   </details>

### 8. Visualizing RRL and IR data

   **Status: New**

   *Learning objective*: Generate informative data visualizations

   *Criteria for success*: Obtain relevant IR data and create IR+RRL visualizations

   <details>
   As described in the telescope proposal, some of our targets have associated SOFIA data. We need to obtain these data and include them in our analysis. This is not urgent, and may require me to reach out to our collaborator, Lars.
   </details>


