import os
import os
import re
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
weather_api_key = os.getenv("OPENWEATHER_API_KEY")
amadeus_api_key = os.getenv("AMADEUS_API_KEY")
amadeus_api_secret = os.getenv("AMADEUS_API_SECRET")

exa_api_key = os.environ.get('EXA_API_KEY')
# Set the environment variables
os.environ['EXA_API_KEY'] = exa_api_key

from textwrap import dedent
from datetime import datetime
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.exa import ExaTools

load_dotenv()

class TravelParamExtractor:
    @staticmethod
    def parse_query(query):
        """Extract travel parameters using regex patterns"""
        params = {
            'origin': None,
            'destination': None,
            'start_date': None,
            'end_date': None,
            'travelers': 1
        }

        # Location extraction
        location_match = re.search(r'(?:visit|go to|travel to) (\w+(?: \w+)?)', query, re.I)
        if location_match:
            params['destination'] = location_match.group(1)

        # Date extraction
        date_pattern = r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(?:to|\-)\s*\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)'
        date_match = re.search(date_pattern, query, re.I)
        if date_match:
            dates = re.findall(r'\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*', date_match.group(0), re.I)
            if len(dates) == 2:
                params['start_date'] = datetime.strptime(dates[0], '%d %b').strftime('%Y-%m-%d')
                params['end_date'] = datetime.strptime(dates[1], '%d %b').strftime('%Y-%m-%d')

        return params

class WeatherTool:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5/forecast"

    def run(self, query):
        """Get weather forecast for any location"""
        try:
            params = TravelParamExtractor.parse_query(query)
            if not params['destination']:
                return "Please specify a destination"
            
            response = requests.get(self.base_url, params={
                "q": params['destination'],
                "appid": self.api_key,
                "units": "metric"
            })
            
            data = response.json()
            forecast = "\n".join([
                f"{datetime.fromtimestamp(item['dt']).strftime('%b %d %H:%M')}: "
                f"{item['main']['temp']}°C, "
                f"{item['weather'][0]['description'].capitalize()}"
                for item in data['list'][:8]  # Next 24 hours
            ])
            
            return f"Weather forecast for {params['destination']}:\n{forecast}"

        except Exception as e:
            return f"Weather data error: {str(e)}"

class FlightSearchTool:
    def __init__(self):
        self.amadeus_api_key = os.getenv("AMADEUS_API_KEY")
        self.amadeus_api_secret = os.getenv("AMADEUS_API_SECRET")
        self.token = self._get_auth_token()

    def _get_auth_token(self):
        auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        response = requests.post(auth_url, data={
            "grant_type": "client_credentials",
            "client_id": self.amadeus_api_key,
            "client_secret": self.amadeus_api_secret
        })
        return response.json().get("access_token")

    def run(self, query):
        """Search flights for any route"""
        try:
            params = TravelParamExtractor.parse_query(query)
            if not all([params['origin'], params['destination'], params['start_date']]):
                return "Missing flight parameters"

            flight_params = {
                "originLocationCode": params['origin'][:3].upper(),
                "destinationLocationCode": params['destination'][:3].upper(),
                "departureDate": params['start_date'],
                "adults": params['travelers'],
                "currencyCode": "USD",
                "max": 5
            }

            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(
                "https://test.api.amadeus.com/v2/shopping/flight-offers",
                headers=headers,
                params=flight_params
            )

            flights = []
            for offer in response.json().get('data', [])[:3]:
                flight = offer['itineraries'][0]['segments'][0]
                price = offer['price']['total']
                flights.append(
                    f"{flight['carrierCode']}{flight['number']} "
                    f"{flight['departure']['iataCode']}→{flight['arrival']['iataCode']} "
                    f"${price}"
                )

            return f"Flights from {params['origin']} to {params['destination']}:\n" + "\n".join(flights)

        except Exception as e:
            return f"Flight search error: {str(e)}"

class GlobalTravelAgent:
    def __init__(self):
        self.agent = Agent(
            name="World Explorer",
            model=Gemini(
                id="gemini-2.0-flash",
                api_key=api_key,
            ),
            tools=[WeatherTool(), FlightSearchTool(), ExaTools()],
            markdown=True,
            description=dedent("""\
                You are Globe Hopper Pro - The Ultimate AI Travel Planning System! ✈️🌐

                Capabilities include:
                ✅ Real-time flight price analysis
                ✅ Hotel quality/cost optimization
                ✅ Weather pattern prediction
                ✅ Cultural event tracking
                ✅ Transportation logistics
                ✅ Safety advisory integration
                ✅ Budget optimization algorithms
                ✅ Accessibility compliance checks"""),
            instructions=dedent("""\
                PLANNING WORKFLOW:

1. INITIAL ANALYSIS 📋
- Detect origin/destination locations
- Parse exact travel dates
- Identify group composition
- Note special requirements

2. ENVIRONMENT SCAN 🌦️
- Check historical weather patterns
- Verify current weather forecasts
- Research local events/festivals
- Review travel advisories

3. TRANSPORTATION PLANNING ✈️
- Compare flight options/prices
- Calculate optimal arrival/departure times
- Evaluate airport transfer options
- Consider ground transportation

4. ACCOMMODATION STRATEGY 🛏️
- Balance location vs price
- Verify amenities/accessibility
- Check recent guest reviews
- Identify promotion deals

5. ACTIVITY OPTIMIZATION 🗺️
- Cluster nearby attractions
- Balance activity types
- Include local experiences
- Plan weather contingencies

6. FINAL PRESENTATION 💼
- Clear day-by-day structure
- Visual timeline mapping
- Budget breakdowns
- Multiple format options"""),
            expected_output=dedent("""\
                # {DESTINATION} TRAVEL BLUEPRINT 🗺️
                **Travel Period:** {dates} | **Travelers:** {group_size} | **Budget Tier:** {budget_level}

                ## 🌦️ Weather Advisory
                {weather_summary_with_packing_tips}

                ## ✈️ Flight Recommendations
                {flight_options_with_prices_and_durations}

                ## 🛎️ Accommodation Picks
                {hotel_options_with_ratings_and_amenities}

                ## 📅 Daily Itinerary
                {hour-by-hour_schedule_with_transit_times}

                ### Day 1: Arrival & Orientation
                - 14:00 Hotel check-in
                - 15:00 Local area orientation
                - 18:00 Welcome dinner at {top_local_restaurant}

                ## 💰 Budget Estimate
                | Category    | Estimated Cost | Notes               |
                |-------------|----------------|---------------------|
                | Flights     | ${flight_cost} | Economy class fares|
                | Accommodation | ${hotel_cost} | 4-star central location |

                ## 🚨 Travel Essentials
                - Required documentation
                - Health precautions
                - Cultural etiquette tips
                - Emergency contacts"""),
            add_datetime_to_instructions=True,
            show_tool_calls=True,
        )

if __name__ == "__main__":
    agent = GlobalTravelAgent()
    query = input("Enter your travel query: ")
    agent.agent.print_response(query, stream=True)