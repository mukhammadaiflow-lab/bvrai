"""
Industry Terminology Module

This module provides industry-specific terminology, definitions,
and customer-friendly explanations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import IndustryType, TermDefinition


# Healthcare Terminology
HEALTHCARE_TERMS: List[TermDefinition] = [
    TermDefinition(
        term="Copay",
        definition="Fixed amount you pay for a covered health care service",
        abbreviation="copay",
        synonyms=["copayment", "co-pay"],
        examples=["Your copay for this visit is $30"],
        customer_friendly="The fixed amount you pay at your visit",
    ),
    TermDefinition(
        term="Deductible",
        definition="Amount you pay before insurance starts covering costs",
        synonyms=["annual deductible"],
        examples=["You've met your deductible for the year"],
        customer_friendly="What you pay out-of-pocket before insurance kicks in",
    ),
    TermDefinition(
        term="Pre-authorization",
        definition="Insurance approval needed before certain procedures",
        abbreviation="prior auth",
        synonyms=["prior authorization", "pre-approval"],
        customer_friendly="Getting your insurance to approve a procedure beforehand",
    ),
    TermDefinition(
        term="Referral",
        definition="Recommendation from primary doctor to see a specialist",
        examples=["You'll need a referral to see the specialist"],
        customer_friendly="A note from your regular doctor to see another doctor",
    ),
    TermDefinition(
        term="Preventive care",
        definition="Health services to prevent illness or detect it early",
        synonyms=["wellness care", "preventative care"],
        examples=["Annual physicals are covered as preventive care"],
        customer_friendly="Check-ups and screenings to keep you healthy",
    ),
    TermDefinition(
        term="Out-of-pocket maximum",
        definition="Maximum amount you pay in a year before insurance covers 100%",
        abbreviation="OOP max",
        synonyms=["maximum out-of-pocket"],
        customer_friendly="The most you'll pay in a year - after that, insurance covers everything",
    ),
    TermDefinition(
        term="PHI",
        definition="Protected Health Information - any health data that can identify you",
        abbreviation="PHI",
        related_terms=["HIPAA", "patient privacy"],
        customer_friendly="Your private health information",
    ),
]


# Dental Terminology
DENTAL_TERMS: List[TermDefinition] = [
    TermDefinition(
        term="Prophylaxis",
        definition="Professional dental cleaning",
        abbreviation="prophy",
        synonyms=["dental cleaning", "teeth cleaning"],
        customer_friendly="A professional teeth cleaning",
    ),
    TermDefinition(
        term="Periodontal",
        definition="Related to the gums and supporting structures of teeth",
        abbreviation="perio",
        related_terms=["gum disease", "periodontitis"],
        customer_friendly="Having to do with your gums",
    ),
    TermDefinition(
        term="Crown",
        definition="Cap placed over a tooth to restore shape and function",
        synonyms=["dental crown", "cap"],
        examples=["The damaged tooth will need a crown"],
        customer_friendly="A protective cap for a damaged tooth",
    ),
    TermDefinition(
        term="Root canal",
        definition="Treatment to save a badly infected or damaged tooth",
        synonyms=["endodontic treatment"],
        customer_friendly="A procedure to save a tooth with infected nerves",
    ),
    TermDefinition(
        term="Scaling and root planing",
        definition="Deep cleaning below the gumline for gum disease",
        abbreviation="SRP",
        synonyms=["deep cleaning"],
        customer_friendly="A deep cleaning to treat gum disease",
    ),
    TermDefinition(
        term="Composite filling",
        definition="Tooth-colored filling material",
        synonyms=["white filling", "resin filling"],
        customer_friendly="A tooth-colored filling that blends with your teeth",
    ),
]


# Legal Terminology
LEGAL_TERMS: List[TermDefinition] = [
    TermDefinition(
        term="Retainer",
        definition="Upfront payment to secure legal services",
        synonyms=["retainer fee"],
        examples=["The retainer covers the initial work on your case"],
        customer_friendly="An upfront payment to hire the attorney",
    ),
    TermDefinition(
        term="Contingency fee",
        definition="Attorney fee paid only if the case is won",
        synonyms=["no win no fee"],
        examples=["We work on a contingency fee basis"],
        customer_friendly="You only pay if we win your case",
    ),
    TermDefinition(
        term="Statute of limitations",
        definition="Time limit to file a legal claim",
        examples=["The statute of limitations for this type of case is two years"],
        customer_friendly="The deadline to file your legal case",
    ),
    TermDefinition(
        term="Discovery",
        definition="Process of gathering evidence before trial",
        related_terms=["depositions", "interrogatories"],
        customer_friendly="The phase where both sides gather evidence",
    ),
    TermDefinition(
        term="Settlement",
        definition="Agreement to resolve a case without trial",
        synonyms=["out-of-court settlement"],
        customer_friendly="Resolving your case without going to court",
    ),
    TermDefinition(
        term="Deposition",
        definition="Sworn testimony given outside of court",
        related_terms=["discovery", "testimony"],
        customer_friendly="Answering questions under oath before the trial",
    ),
    TermDefinition(
        term="Plaintiff",
        definition="Person who brings a lawsuit",
        synonyms=["complainant", "claimant"],
        customer_friendly="The person filing the lawsuit",
    ),
    TermDefinition(
        term="Defendant",
        definition="Person being sued or accused",
        customer_friendly="The person being sued",
    ),
]


# Real Estate Terminology
REAL_ESTATE_TERMS: List[TermDefinition] = [
    TermDefinition(
        term="Escrow",
        definition="Neutral third party holding funds during a transaction",
        examples=["Your earnest money will be held in escrow"],
        customer_friendly="A secure holding account for your deposit",
    ),
    TermDefinition(
        term="Earnest money",
        definition="Deposit showing buyer's commitment to purchase",
        synonyms=["good faith deposit", "earnest money deposit"],
        abbreviation="EMD",
        customer_friendly="Your deposit showing you're serious about buying",
    ),
    TermDefinition(
        term="Pre-approval",
        definition="Lender's conditional commitment to loan a specific amount",
        synonyms=["mortgage pre-approval"],
        related_terms=["pre-qualification"],
        customer_friendly="A lender saying they'll likely approve your loan",
    ),
    TermDefinition(
        term="Closing costs",
        definition="Fees paid at the end of a real estate transaction",
        examples=["Closing costs are typically 2-5% of the purchase price"],
        customer_friendly="Additional fees paid when finalizing your purchase",
    ),
    TermDefinition(
        term="Contingency",
        definition="Condition that must be met for the sale to proceed",
        examples=["The inspection contingency allows you to negotiate repairs"],
        customer_friendly="A condition that protects you during the purchase",
    ),
    TermDefinition(
        term="Appraisal",
        definition="Professional estimate of a property's value",
        related_terms=["home value", "market value"],
        customer_friendly="An expert's opinion of what the home is worth",
    ),
    TermDefinition(
        term="Title insurance",
        definition="Insurance protecting against title defects",
        customer_friendly="Protection against problems with property ownership",
    ),
    TermDefinition(
        term="CMA",
        definition="Comparative Market Analysis - estimate of home value based on similar sales",
        abbreviation="CMA",
        synonyms=["market analysis", "comp analysis"],
        customer_friendly="An estimate of your home's value based on similar homes that sold",
    ),
]


# HVAC Terminology
HVAC_TERMS: List[TermDefinition] = [
    TermDefinition(
        term="SEER",
        definition="Seasonal Energy Efficiency Ratio - measures AC efficiency",
        abbreviation="SEER",
        examples=["A higher SEER rating means lower energy bills"],
        customer_friendly="A rating showing how efficient your AC is - higher is better",
    ),
    TermDefinition(
        term="BTU",
        definition="British Thermal Unit - measure of heating/cooling capacity",
        abbreviation="BTU",
        customer_friendly="A measure of heating or cooling power",
    ),
    TermDefinition(
        term="Tonnage",
        definition="Cooling capacity measurement (12,000 BTU = 1 ton)",
        synonyms=["tons", "AC tonnage"],
        examples=["Your home needs a 3-ton unit"],
        customer_friendly="The size/power of your AC system",
    ),
    TermDefinition(
        term="Heat pump",
        definition="System that heats and cools by transferring heat",
        related_terms=["HVAC", "AC"],
        customer_friendly="A system that can both heat and cool your home efficiently",
    ),
    TermDefinition(
        term="Condenser",
        definition="Outdoor unit that releases or collects heat",
        synonyms=["outdoor unit", "compressor unit"],
        customer_friendly="The outdoor part of your AC system",
    ),
    TermDefinition(
        term="Evaporator coil",
        definition="Indoor coil that absorbs heat from the air",
        synonyms=["indoor coil", "A-coil"],
        customer_friendly="The indoor part that cools the air",
    ),
    TermDefinition(
        term="Refrigerant",
        definition="Chemical that absorbs and releases heat in AC systems",
        synonyms=["freon", "R-410A"],
        customer_friendly="The cooling fluid inside your AC system",
    ),
    TermDefinition(
        term="Ductwork",
        definition="System of tubes distributing heated/cooled air",
        synonyms=["ducts", "air ducts"],
        customer_friendly="The tubes that carry air throughout your home",
    ),
]


# Insurance Terminology
INSURANCE_TERMS: List[TermDefinition] = [
    TermDefinition(
        term="Premium",
        definition="Amount paid for insurance coverage",
        synonyms=["insurance premium"],
        examples=["Your monthly premium is $150"],
        customer_friendly="Your regular payment for insurance coverage",
    ),
    TermDefinition(
        term="Deductible",
        definition="Amount you pay before insurance covers a claim",
        examples=["You have a $500 deductible on your auto policy"],
        customer_friendly="What you pay out-of-pocket before insurance pays",
    ),
    TermDefinition(
        term="Liability coverage",
        definition="Protection when you're responsible for damage to others",
        related_terms=["bodily injury", "property damage"],
        customer_friendly="Coverage that pays if you cause damage to someone else",
    ),
    TermDefinition(
        term="Comprehensive coverage",
        definition="Coverage for non-collision damage to your vehicle",
        synonyms=["comp coverage", "other than collision"],
        examples=["Comprehensive covers theft, weather damage, and vandalism"],
        customer_friendly="Coverage for damage that isn't from an accident",
    ),
    TermDefinition(
        term="Collision coverage",
        definition="Coverage for damage to your vehicle in an accident",
        customer_friendly="Coverage for damage to your car in a crash",
    ),
    TermDefinition(
        term="Endorsement",
        definition="Addition or modification to an insurance policy",
        synonyms=["rider", "policy rider"],
        customer_friendly="An add-on to customize your policy",
    ),
    TermDefinition(
        term="Underwriting",
        definition="Process of evaluating risk to determine coverage and price",
        examples=["The underwriting process takes 1-2 weeks"],
        customer_friendly="How the insurance company decides your rate",
    ),
    TermDefinition(
        term="Claim",
        definition="Request for payment from an insurance policy",
        synonyms=["insurance claim", "file a claim"],
        customer_friendly="Asking your insurance to pay for a covered loss",
    ),
]


# Financial Services Terminology
FINANCIAL_TERMS: List[TermDefinition] = [
    TermDefinition(
        term="Asset allocation",
        definition="How investments are divided among different asset classes",
        related_terms=["diversification", "portfolio"],
        customer_friendly="How your money is spread across different investments",
    ),
    TermDefinition(
        term="Diversification",
        definition="Spreading investments to reduce risk",
        related_terms=["asset allocation", "risk management"],
        customer_friendly="Not putting all your eggs in one basket",
    ),
    TermDefinition(
        term="Fiduciary",
        definition="Legally required to act in client's best interest",
        synonyms=["fiduciary duty"],
        customer_friendly="An advisor who must put your interests first",
    ),
    TermDefinition(
        term="IRA",
        definition="Individual Retirement Account - tax-advantaged retirement savings",
        abbreviation="IRA",
        related_terms=["Roth IRA", "Traditional IRA", "401k"],
        customer_friendly="A special account for saving for retirement with tax benefits",
    ),
    TermDefinition(
        term="401(k)",
        definition="Employer-sponsored retirement savings plan",
        synonyms=["401k"],
        related_terms=["IRA", "retirement plan"],
        customer_friendly="Retirement savings through your employer",
    ),
    TermDefinition(
        term="Expense ratio",
        definition="Annual cost of owning a mutual fund or ETF",
        examples=["This fund has a low 0.05% expense ratio"],
        customer_friendly="The annual fee charged by an investment fund",
    ),
    TermDefinition(
        term="Risk tolerance",
        definition="Ability and willingness to accept investment losses",
        related_terms=["risk appetite", "investment horizon"],
        customer_friendly="How comfortable you are with investment ups and downs",
    ),
]


# Plumbing Terminology
PLUMBING_TERMS: List[TermDefinition] = [
    TermDefinition(
        term="Main shutoff",
        definition="Valve that controls water to entire property",
        synonyms=["main valve", "water main shutoff"],
        customer_friendly="The valve that turns off water to your whole house",
    ),
    TermDefinition(
        term="Trap",
        definition="U-shaped pipe that holds water to block sewer gases",
        synonyms=["P-trap", "drain trap"],
        customer_friendly="The curved pipe under sinks that prevents bad smells",
    ),
    TermDefinition(
        term="Tankless water heater",
        definition="Water heater that heats water on demand without a storage tank",
        synonyms=["on-demand water heater", "instant water heater"],
        customer_friendly="A water heater that heats water instantly when you need it",
    ),
    TermDefinition(
        term="Sump pump",
        definition="Pump that removes water from basement or crawl space",
        related_terms=["basement drainage", "flood prevention"],
        customer_friendly="A pump that keeps your basement dry",
    ),
    TermDefinition(
        term="Backflow preventer",
        definition="Device preventing contaminated water from flowing backward",
        synonyms=["backflow device"],
        customer_friendly="A safety device that keeps dirty water out of your clean water",
    ),
    TermDefinition(
        term="Camera inspection",
        definition="Video inspection of pipes using a specialized camera",
        synonyms=["pipe camera", "sewer camera"],
        customer_friendly="Using a camera to see inside your pipes",
    ),
]


# Restaurant Terminology
RESTAURANT_TERMS: List[TermDefinition] = [
    TermDefinition(
        term="Prix fixe",
        definition="Set menu with fixed price",
        synonyms=["fixed price menu", "set menu"],
        customer_friendly="A special menu with a set price for multiple courses",
    ),
    TermDefinition(
        term="Tasting menu",
        definition="Multi-course menu showcasing chef's selections",
        synonyms=["degustation menu"],
        customer_friendly="A special menu where the chef picks the courses for you",
    ),
    TermDefinition(
        term="Table d'hôte",
        definition="Complete meal at a fixed price",
        customer_friendly="A complete meal at one price",
    ),
    TermDefinition(
        term="BYO",
        definition="Bring Your Own - typically referring to wine/alcohol",
        abbreviation="BYO",
        synonyms=["BYOB"],
        customer_friendly="You can bring your own wine or alcohol",
    ),
    TermDefinition(
        term="Corkage fee",
        definition="Fee charged to open wine brought by customer",
        customer_friendly="A fee if you bring your own wine",
    ),
]


# Auto Dealership Terminology
AUTO_TERMS: List[TermDefinition] = [
    TermDefinition(
        term="MSRP",
        definition="Manufacturer's Suggested Retail Price",
        abbreviation="MSRP",
        synonyms=["sticker price", "list price"],
        customer_friendly="The price the manufacturer suggests for the vehicle",
    ),
    TermDefinition(
        term="Invoice price",
        definition="Price the dealer paid for the vehicle",
        synonyms=["dealer cost"],
        customer_friendly="What the dealer paid for the car",
    ),
    TermDefinition(
        term="APR",
        definition="Annual Percentage Rate - cost of financing",
        abbreviation="APR",
        examples=["We can offer 2.9% APR financing"],
        customer_friendly="The interest rate for your car loan",
    ),
    TermDefinition(
        term="Residual value",
        definition="Estimated value at end of lease term",
        related_terms=["lease", "depreciation"],
        customer_friendly="What the car is expected to be worth at the end of your lease",
    ),
    TermDefinition(
        term="Trade-in value",
        definition="Amount offered for your current vehicle",
        synonyms=["trade value", "trade-in"],
        customer_friendly="What we'll give you for your current car",
    ),
    TermDefinition(
        term="Gap insurance",
        definition="Insurance covering difference between car value and loan balance",
        synonyms=["gap coverage"],
        customer_friendly="Insurance that covers you if your car is totaled and you owe more than it's worth",
    ),
    TermDefinition(
        term="CPO",
        definition="Certified Pre-Owned - used vehicle certified by manufacturer",
        abbreviation="CPO",
        synonyms=["certified used"],
        customer_friendly="A used car that's been inspected and certified",
    ),
]


# Registry of terminology by industry
TERMINOLOGY_REGISTRY: Dict[IndustryType, List[TermDefinition]] = {
    IndustryType.HEALTHCARE_GENERAL: HEALTHCARE_TERMS,
    IndustryType.HEALTHCARE_DENTAL: DENTAL_TERMS,
    IndustryType.LEGAL: LEGAL_TERMS,
    IndustryType.REAL_ESTATE: REAL_ESTATE_TERMS,
    IndustryType.HVAC: HVAC_TERMS,
    IndustryType.INSURANCE: INSURANCE_TERMS,
    IndustryType.FINANCIAL_SERVICES: FINANCIAL_TERMS,
    IndustryType.PLUMBING: PLUMBING_TERMS,
    IndustryType.RESTAURANT: RESTAURANT_TERMS,
    IndustryType.AUTO_DEALERSHIP: AUTO_TERMS,
}


class TerminologyManager:
    """
    Manages industry terminology and provides lookup functionality.

    Features:
    - Term lookup by industry
    - Abbreviation expansion
    - Customer-friendly explanations
    - Related term suggestions
    """

    def __init__(self):
        """Initialize terminology manager."""
        self._term_index: Dict[str, List[TermDefinition]] = {}
        self._abbreviation_index: Dict[str, TermDefinition] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        """Build search indices for terms."""
        for terms in TERMINOLOGY_REGISTRY.values():
            for term_def in terms:
                # Index by term
                term_lower = term_def.term.lower()
                if term_lower not in self._term_index:
                    self._term_index[term_lower] = []
                self._term_index[term_lower].append(term_def)

                # Index by synonyms
                for synonym in term_def.synonyms:
                    syn_lower = synonym.lower()
                    if syn_lower not in self._term_index:
                        self._term_index[syn_lower] = []
                    self._term_index[syn_lower].append(term_def)

                # Index by abbreviation
                if term_def.abbreviation:
                    self._abbreviation_index[term_def.abbreviation.lower()] = term_def

    def get_terms_for_industry(
        self,
        industry_type: IndustryType,
    ) -> List[TermDefinition]:
        """Get all terms for an industry."""
        return TERMINOLOGY_REGISTRY.get(industry_type, [])

    def lookup_term(
        self,
        term: str,
        industry_type: Optional[IndustryType] = None,
    ) -> Optional[TermDefinition]:
        """
        Look up a term definition.

        Args:
            term: Term to look up
            industry_type: Limit to specific industry

        Returns:
            Term definition if found
        """
        term_lower = term.lower()

        # Check abbreviations first
        if term_lower in self._abbreviation_index:
            result = self._abbreviation_index[term_lower]
            if industry_type:
                industry_terms = TERMINOLOGY_REGISTRY.get(industry_type, [])
                if result in industry_terms:
                    return result
            else:
                return result

        # Check term index
        matches = self._term_index.get(term_lower, [])

        if industry_type:
            industry_terms = TERMINOLOGY_REGISTRY.get(industry_type, [])
            for match in matches:
                if match in industry_terms:
                    return match
            return None

        return matches[0] if matches else None

    def expand_abbreviation(self, abbrev: str) -> Optional[str]:
        """
        Expand an abbreviation to full term.

        Args:
            abbrev: Abbreviation to expand

        Returns:
            Full term if found
        """
        term_def = self._abbreviation_index.get(abbrev.lower())
        return term_def.term if term_def else None

    def get_customer_explanation(
        self,
        term: str,
        industry_type: Optional[IndustryType] = None,
    ) -> Optional[str]:
        """
        Get customer-friendly explanation for a term.

        Args:
            term: Term to explain
            industry_type: Industry context

        Returns:
            Customer-friendly explanation
        """
        term_def = self.lookup_term(term, industry_type)
        if term_def:
            return term_def.customer_friendly or term_def.definition
        return None

    def find_related_terms(
        self,
        term: str,
        industry_type: Optional[IndustryType] = None,
    ) -> List[str]:
        """
        Find terms related to a given term.

        Args:
            term: Term to find relations for
            industry_type: Industry context

        Returns:
            List of related terms
        """
        term_def = self.lookup_term(term, industry_type)
        if term_def:
            return term_def.related_terms
        return []

    def search_terms(
        self,
        query: str,
        industry_type: Optional[IndustryType] = None,
    ) -> List[TermDefinition]:
        """
        Search for terms matching a query.

        Args:
            query: Search query
            industry_type: Limit to specific industry

        Returns:
            Matching term definitions
        """
        query_lower = query.lower()
        results = []
        seen_terms = set()

        terms_to_search = (
            TERMINOLOGY_REGISTRY.get(industry_type, [])
            if industry_type
            else [t for terms in TERMINOLOGY_REGISTRY.values() for t in terms]
        )

        for term_def in terms_to_search:
            if term_def.term in seen_terms:
                continue

            # Check term
            if query_lower in term_def.term.lower():
                results.append(term_def)
                seen_terms.add(term_def.term)
                continue

            # Check definition
            if query_lower in term_def.definition.lower():
                results.append(term_def)
                seen_terms.add(term_def.term)
                continue

            # Check synonyms
            for synonym in term_def.synonyms:
                if query_lower in synonym.lower():
                    results.append(term_def)
                    seen_terms.add(term_def.term)
                    break

        return results


def get_industry_terminology(industry_type: IndustryType) -> List[TermDefinition]:
    """Get terminology for an industry."""
    return TERMINOLOGY_REGISTRY.get(industry_type, [])


def create_terminology_manager() -> TerminologyManager:
    """Create a terminology manager instance."""
    return TerminologyManager()


__all__ = [
    "TerminologyManager",
    "TERMINOLOGY_REGISTRY",
    "get_industry_terminology",
    "create_terminology_manager",
    "HEALTHCARE_TERMS",
    "DENTAL_TERMS",
    "LEGAL_TERMS",
    "REAL_ESTATE_TERMS",
    "HVAC_TERMS",
    "INSURANCE_TERMS",
    "FINANCIAL_TERMS",
    "PLUMBING_TERMS",
    "RESTAURANT_TERMS",
    "AUTO_TERMS",
]
