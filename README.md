<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/cudash/cudash">
  </a>

<h3 align="center">CUDASH</h3>

  <p align="center">
    A historical and interactive dashboard with 8 years of preloaded Con Edison data and 1 year of Building Managment System (BMS) data for 41 Cooper Square (41CS) which is apart of the Cooper Union. The Con Edision data also includes the CO2 emissions for the Residence Hall and Foundation Building at Cooper Union. 
    <br />

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#public-repository">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#cloning-this-github">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



## Built With Python, DashApp, Render, Docker, and DashTools
     

<!-- GETTING STARTED -->
## Getting Started

Create DashApp framework through using DashTools: 
When utilizing DashTools, it will create an app template with a default README.md file that has the below information on how to deploy the app locally on your machine: 
    
Welcome to your [Plotly Dash](https://plotly.com/dash/) App! This is a template for your myapp app.
See the [Dash Documentation](https://dash.plotly.com/introduction) for more information on how to get your app up and running.

Run `src/app.py` and navigate to http://127.0.0.1:8050/ in your browser.
    
DashTools and Render work together since Dashtools seamlessly creates the all the required files for deployment. Here is the Render dashboard snippet where you can choose which commited version of the app to deploy: ![image](https://github.com/cudash/cudash/assets/130943510/0494b624-ee97-42fc-8272-b04b8cf090ce)
and here is the snippet of the generated files that DashTools generates ![image](https://github.com/cudash/cudash/assets/130943510/47d79c83-3aff-483e-8e15-6bd677dd3434)
Ultimately, the free plan in Render was not enough for the task so Gary Kim is hosting on his server and has included the required Docker files to launch the app to this github. 

### Public repository
These are steps that were used to push changes into the public repo:

   ```sh
   git init
   ```
   ```sh
   git add
   ```
   ```sh
   git commit m "a message"
   ```
   ```sh
   git push
   ```
    
### Cloning this Github Repository
    Please refer to the below docs if you want to clone the github to the document: 
    https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository
<!-- USAGE EXAMPLES -->
## Usage
Requires dashboard updates to keep running optimally:
  * Each year add data for each BMS point located in the data file from Dec 7 from the prior year to current and update data file
  * Each year update the carbon coefficients and 
Chiller Plant Efficency
![image](https://github.com/cudash/cudash/assets/130943510/26af3ed0-71df-41a3-8590-ea47a2064828)
Air Handler Systems
![image](https://github.com/cudash/cudash/assets/130943510/1b5f8045-ae92-4042-abcb-968666dbc866)
Cooling Tower Model 
![image](https://github.com/cudash/cudash/assets/130943510/7af27e6c-5dec-4eec-ba81-b29022b19103)

_For more examples, please refer to the link: https://cudash.garykim.dev/_


<!-- CONTRIBUTING -->
## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- CONTACT -->
## Contact

CUDASH - cuenergydashboard23@gmail.com

Project Link:
  
    Render (currently not in use): https://cudash.onrender.com 
    Please Note: If there is an upgrade in the Render plan used, could be a viable solution
  
    Docker: https://cudash.garykim.dev/ (Docker requirements have been added into the repo)


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments for Github

* DashTools
* Gary Kim
* Eric Leong 
*https://github.com/othneildrew/Best-README-Template/ for guidance on ReadME

