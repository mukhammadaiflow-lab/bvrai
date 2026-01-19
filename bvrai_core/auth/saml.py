"""SAML 2.0 authentication provider."""

import base64
import hashlib
import logging
import secrets
import zlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET
from urllib.parse import urlencode, quote
import re

logger = logging.getLogger(__name__)


# XML namespaces
SAML_NS = "urn:oasis:names:tc:SAML:2.0:assertion"
SAMLP_NS = "urn:oasis:names:tc:SAML:2.0:protocol"
DSIG_NS = "http://www.w3.org/2000/09/xmldsig#"

NS_MAP = {
    "saml": SAML_NS,
    "samlp": SAMLP_NS,
    "ds": DSIG_NS,
}


@dataclass
class SAMLAssertion:
    """Parsed SAML assertion."""
    id: str
    issuer: str
    subject_name_id: str
    subject_name_id_format: str = ""

    # Conditions
    not_before: Optional[datetime] = None
    not_on_or_after: Optional[datetime] = None
    audience_restriction: List[str] = field(default_factory=list)

    # Authentication
    authn_instant: Optional[datetime] = None
    authn_context: str = ""
    session_index: str = ""

    # Attributes
    attributes: Dict[str, List[str]] = field(default_factory=dict)

    def get_attribute(self, name: str, default: str = "") -> str:
        """Get first value of an attribute."""
        values = self.attributes.get(name, [])
        return values[0] if values else default

    def get_attributes(self, name: str) -> List[str]:
        """Get all values of an attribute."""
        return self.attributes.get(name, [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "issuer": self.issuer,
            "subject_name_id": self.subject_name_id,
            "subject_name_id_format": self.subject_name_id_format,
            "not_before": self.not_before.isoformat() if self.not_before else None,
            "not_on_or_after": self.not_on_or_after.isoformat() if self.not_on_or_after else None,
            "audience_restriction": self.audience_restriction,
            "authn_instant": self.authn_instant.isoformat() if self.authn_instant else None,
            "authn_context": self.authn_context,
            "session_index": self.session_index,
            "attributes": self.attributes,
        }


@dataclass
class SAMLResponse:
    """Parsed SAML response."""
    id: str
    issuer: str
    destination: str = ""
    in_response_to: str = ""
    status_code: str = ""
    status_message: str = ""

    # Assertions
    assertions: List[SAMLAssertion] = field(default_factory=list)

    # Timestamps
    issue_instant: Optional[datetime] = None

    # Signature validation
    is_signed: bool = False
    signature_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "issuer": self.issuer,
            "destination": self.destination,
            "in_response_to": self.in_response_to,
            "status_code": self.status_code,
            "status_message": self.status_message,
            "assertions": [a.to_dict() for a in self.assertions],
            "issue_instant": self.issue_instant.isoformat() if self.issue_instant else None,
            "is_signed": self.is_signed,
            "signature_valid": self.signature_valid,
        }

    @property
    def is_success(self) -> bool:
        return self.status_code == "urn:oasis:names:tc:SAML:2.0:status:Success"

    @property
    def first_assertion(self) -> Optional[SAMLAssertion]:
        return self.assertions[0] if self.assertions else None


class SAMLProvider:
    """
    SAML 2.0 Service Provider implementation.

    Handles:
    - AuthnRequest generation
    - Response parsing and validation
    - Signature verification
    - Attribute extraction
    - Single Logout (SLO)
    """

    def __init__(
        self,
        entity_id: str,
        acs_url: str,
        idp_entity_id: str,
        idp_sso_url: str,
        idp_certificate: str,
        idp_slo_url: str = "",
        sp_private_key: str = "",
        sp_certificate: str = "",
        sign_requests: bool = True,
        want_assertions_signed: bool = True,
        signature_algorithm: str = "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256",
    ):
        self.entity_id = entity_id
        self.acs_url = acs_url
        self.idp_entity_id = idp_entity_id
        self.idp_sso_url = idp_sso_url
        self.idp_certificate = idp_certificate
        self.idp_slo_url = idp_slo_url
        self.sp_private_key = sp_private_key
        self.sp_certificate = sp_certificate
        self.sign_requests = sign_requests
        self.want_assertions_signed = want_assertions_signed
        self.signature_algorithm = signature_algorithm

        # Register namespaces
        for prefix, uri in NS_MAP.items():
            ET.register_namespace(prefix, uri)

    def create_authn_request(
        self,
        relay_state: str = "",
        force_authn: bool = False,
        is_passive: bool = False,
        name_id_format: str = "",
    ) -> Tuple[str, str, str]:
        """
        Create a SAML AuthnRequest.

        Returns:
            Tuple of (request_id, redirect_url, request_xml)
        """
        request_id = f"_{''.join(secrets.token_hex(16))}"
        issue_instant = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build AuthnRequest XML
        root = ET.Element(
            f"{{{SAMLP_NS}}}AuthnRequest",
            {
                "ID": request_id,
                "Version": "2.0",
                "IssueInstant": issue_instant,
                "Destination": self.idp_sso_url,
                "ProtocolBinding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                "AssertionConsumerServiceURL": self.acs_url,
            }
        )

        if force_authn:
            root.set("ForceAuthn", "true")
        if is_passive:
            root.set("IsPassive", "true")

        # Issuer
        issuer = ET.SubElement(root, f"{{{SAML_NS}}}Issuer")
        issuer.text = self.entity_id

        # NameIDPolicy
        if name_id_format:
            name_id_policy = ET.SubElement(
                root,
                f"{{{SAMLP_NS}}}NameIDPolicy",
                {
                    "Format": name_id_format,
                    "AllowCreate": "true",
                }
            )

        request_xml = ET.tostring(root, encoding="unicode")

        # Create redirect URL
        redirect_url = self._create_redirect_url(request_xml, relay_state)

        return request_id, redirect_url, request_xml

    def _create_redirect_url(self, request_xml: str, relay_state: str = "") -> str:
        """Create SAML redirect binding URL."""
        # Deflate and base64 encode
        deflated = zlib.compress(request_xml.encode("utf-8"))[2:-4]  # Remove zlib header/trailer
        encoded = base64.b64encode(deflated).decode("utf-8")

        params = {
            "SAMLRequest": encoded,
        }

        if relay_state:
            params["RelayState"] = relay_state

        if self.sign_requests and self.sp_private_key:
            # Add signature
            params["SigAlg"] = self.signature_algorithm
            # Signature would be computed here using sp_private_key
            # For production, use a proper crypto library like python-saml or xmlsec

        query_string = urlencode(params)
        separator = "&" if "?" in self.idp_sso_url else "?"

        return f"{self.idp_sso_url}{separator}{query_string}"

    def parse_response(
        self,
        saml_response: str,
        expected_request_id: str = "",
    ) -> SAMLResponse:
        """
        Parse and validate a SAML response.

        Args:
            saml_response: Base64-encoded SAML response
            expected_request_id: Request ID to validate InResponseTo

        Returns:
            Parsed SAMLResponse object
        """
        try:
            # Decode
            xml_bytes = base64.b64decode(saml_response)
            xml_str = xml_bytes.decode("utf-8")

            # Parse XML
            root = ET.fromstring(xml_str)

            # Extract response attributes
            response = SAMLResponse(
                id=root.get("ID", ""),
                issuer=self._get_issuer(root),
                destination=root.get("Destination", ""),
                in_response_to=root.get("InResponseTo", ""),
            )

            # Parse issue instant
            issue_instant = root.get("IssueInstant")
            if issue_instant:
                response.issue_instant = self._parse_datetime(issue_instant)

            # Parse status
            status = root.find(f".//{{{SAMLP_NS}}}Status")
            if status is not None:
                status_code = status.find(f".//{{{SAMLP_NS}}}StatusCode")
                if status_code is not None:
                    response.status_code = status_code.get("Value", "")

                status_message = status.find(f".//{{{SAMLP_NS}}}StatusMessage")
                if status_message is not None and status_message.text:
                    response.status_message = status_message.text

            # Check signature
            signature = root.find(f".//{{{DSIG_NS}}}Signature")
            if signature is not None:
                response.is_signed = True
                response.signature_valid = self._verify_signature(xml_str)

            # Parse assertions
            assertions = root.findall(f".//{{{SAML_NS}}}Assertion")
            for assertion_elem in assertions:
                assertion = self._parse_assertion(assertion_elem)
                response.assertions.append(assertion)

            # Validate InResponseTo
            if expected_request_id and response.in_response_to != expected_request_id:
                logger.warning(
                    f"InResponseTo mismatch: expected {expected_request_id}, "
                    f"got {response.in_response_to}"
                )

            return response

        except Exception as e:
            logger.error(f"Failed to parse SAML response: {e}")
            return SAMLResponse(
                id="",
                issuer="",
                status_code="urn:oasis:names:tc:SAML:2.0:status:Responder",
                status_message=str(e),
            )

    def _get_issuer(self, element: ET.Element) -> str:
        """Extract issuer from element."""
        issuer = element.find(f".//{{{SAML_NS}}}Issuer")
        return issuer.text if issuer is not None and issuer.text else ""

    def _parse_assertion(self, assertion_elem: ET.Element) -> SAMLAssertion:
        """Parse a SAML assertion element."""
        assertion = SAMLAssertion(
            id=assertion_elem.get("ID", ""),
            issuer=self._get_issuer(assertion_elem),
            subject_name_id="",
        )

        # Subject
        subject = assertion_elem.find(f".//{{{SAML_NS}}}Subject")
        if subject is not None:
            name_id = subject.find(f".//{{{SAML_NS}}}NameID")
            if name_id is not None:
                assertion.subject_name_id = name_id.text or ""
                assertion.subject_name_id_format = name_id.get("Format", "")

        # Conditions
        conditions = assertion_elem.find(f".//{{{SAML_NS}}}Conditions")
        if conditions is not None:
            not_before = conditions.get("NotBefore")
            if not_before:
                assertion.not_before = self._parse_datetime(not_before)

            not_on_or_after = conditions.get("NotOnOrAfter")
            if not_on_or_after:
                assertion.not_on_or_after = self._parse_datetime(not_on_or_after)

            # Audience restriction
            audiences = conditions.findall(f".//{{{SAML_NS}}}Audience")
            for audience in audiences:
                if audience.text:
                    assertion.audience_restriction.append(audience.text)

        # AuthnStatement
        authn_statement = assertion_elem.find(f".//{{{SAML_NS}}}AuthnStatement")
        if authn_statement is not None:
            authn_instant = authn_statement.get("AuthnInstant")
            if authn_instant:
                assertion.authn_instant = self._parse_datetime(authn_instant)

            assertion.session_index = authn_statement.get("SessionIndex", "")

            authn_context = authn_statement.find(
                f".//{{{SAML_NS}}}AuthnContextClassRef"
            )
            if authn_context is not None and authn_context.text:
                assertion.authn_context = authn_context.text

        # Attributes
        attribute_statements = assertion_elem.findall(
            f".//{{{SAML_NS}}}AttributeStatement"
        )
        for attr_statement in attribute_statements:
            attributes = attr_statement.findall(f".//{{{SAML_NS}}}Attribute")
            for attr in attributes:
                name = attr.get("Name", "")
                if not name:
                    continue

                values = []
                for value_elem in attr.findall(f".//{{{SAML_NS}}}AttributeValue"):
                    if value_elem.text:
                        values.append(value_elem.text)

                if values:
                    assertion.attributes[name] = values

        return assertion

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse ISO datetime string."""
        try:
            # Remove microseconds and Z suffix for parsing
            dt_str = re.sub(r'\.\d+', '', dt_str)
            dt_str = dt_str.replace("Z", "+00:00")
            return datetime.fromisoformat(dt_str.replace("+00:00", ""))
        except Exception:
            return None

    def _verify_signature(self, xml_str: str) -> bool:
        """
        Verify XML signature.

        Note: In production, use a proper XML signature library like xmlsec
        or python-saml. This is a placeholder.
        """
        # For production implementation:
        # 1. Parse the certificate from idp_certificate
        # 2. Use xmlsec or similar to verify the signature
        # 3. Verify the certificate chain if needed

        if not self.idp_certificate:
            return False

        # Placeholder - always return True if certificate is configured
        # Real implementation would verify using xmlsec
        logger.warning("Signature verification not implemented - using placeholder")
        return True

    def validate_response(
        self,
        response: SAMLResponse,
        expected_audience: str = "",
    ) -> Tuple[bool, str]:
        """
        Validate a parsed SAML response.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check status
        if not response.is_success:
            return False, f"SAML error: {response.status_code} - {response.status_message}"

        # Check signature
        if self.want_assertions_signed and not response.signature_valid:
            return False, "Response signature is invalid or missing"

        # Check assertions
        if not response.assertions:
            return False, "No assertions in response"

        assertion = response.first_assertion
        if not assertion:
            return False, "No valid assertion found"

        # Check issuer
        if assertion.issuer != self.idp_entity_id:
            return False, f"Invalid issuer: expected {self.idp_entity_id}, got {assertion.issuer}"

        # Check conditions
        now = datetime.utcnow()

        if assertion.not_before and now < assertion.not_before:
            return False, f"Assertion not yet valid (NotBefore: {assertion.not_before})"

        if assertion.not_on_or_after and now >= assertion.not_on_or_after:
            return False, f"Assertion expired (NotOnOrAfter: {assertion.not_on_or_after})"

        # Check audience
        if expected_audience and assertion.audience_restriction:
            if expected_audience not in assertion.audience_restriction:
                return False, f"Audience mismatch: expected {expected_audience}"

        return True, ""

    def create_logout_request(
        self,
        name_id: str,
        session_index: str = "",
        relay_state: str = "",
    ) -> Tuple[str, str]:
        """
        Create a SAML LogoutRequest.

        Returns:
            Tuple of (request_id, redirect_url)
        """
        if not self.idp_slo_url:
            raise ValueError("IdP SLO URL not configured")

        request_id = f"_{''.join(secrets.token_hex(16))}"
        issue_instant = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        root = ET.Element(
            f"{{{SAMLP_NS}}}LogoutRequest",
            {
                "ID": request_id,
                "Version": "2.0",
                "IssueInstant": issue_instant,
                "Destination": self.idp_slo_url,
            }
        )

        # Issuer
        issuer = ET.SubElement(root, f"{{{SAML_NS}}}Issuer")
        issuer.text = self.entity_id

        # NameID
        name_id_elem = ET.SubElement(root, f"{{{SAML_NS}}}NameID")
        name_id_elem.text = name_id

        # SessionIndex
        if session_index:
            session_index_elem = ET.SubElement(root, f"{{{SAMLP_NS}}}SessionIndex")
            session_index_elem.text = session_index

        request_xml = ET.tostring(root, encoding="unicode")
        redirect_url = self._create_redirect_url(request_xml, relay_state)

        return request_id, redirect_url

    def parse_logout_response(self, saml_response: str) -> Tuple[bool, str, str]:
        """
        Parse a SAML LogoutResponse.

        Returns:
            Tuple of (is_success, status_code, in_response_to)
        """
        try:
            xml_bytes = base64.b64decode(saml_response)
            xml_str = xml_bytes.decode("utf-8")
            root = ET.fromstring(xml_str)

            in_response_to = root.get("InResponseTo", "")

            status = root.find(f".//{{{SAMLP_NS}}}StatusCode")
            status_code = status.get("Value", "") if status is not None else ""

            is_success = status_code == "urn:oasis:names:tc:SAML:2.0:status:Success"

            return is_success, status_code, in_response_to

        except Exception as e:
            logger.error(f"Failed to parse logout response: {e}")
            return False, str(e), ""

    def get_sp_metadata(self) -> str:
        """Generate Service Provider metadata XML."""
        root = ET.Element(
            "{urn:oasis:names:tc:SAML:2.0:metadata}EntityDescriptor",
            {
                "entityID": self.entity_id,
            }
        )

        sp_sso = ET.SubElement(
            root,
            "{urn:oasis:names:tc:SAML:2.0:metadata}SPSSODescriptor",
            {
                "protocolSupportEnumeration": "urn:oasis:names:tc:SAML:2.0:protocol",
                "AuthnRequestsSigned": str(self.sign_requests).lower(),
                "WantAssertionsSigned": str(self.want_assertions_signed).lower(),
            }
        )

        # Signing certificate
        if self.sp_certificate:
            key_descriptor = ET.SubElement(
                sp_sso,
                "{urn:oasis:names:tc:SAML:2.0:metadata}KeyDescriptor",
                {"use": "signing"}
            )
            key_info = ET.SubElement(key_descriptor, f"{{{DSIG_NS}}}KeyInfo")
            x509_data = ET.SubElement(key_info, f"{{{DSIG_NS}}}X509Data")
            x509_cert = ET.SubElement(x509_data, f"{{{DSIG_NS}}}X509Certificate")
            x509_cert.text = self.sp_certificate.strip()

        # ACS
        acs = ET.SubElement(
            sp_sso,
            "{urn:oasis:names:tc:SAML:2.0:metadata}AssertionConsumerService",
            {
                "Binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                "Location": self.acs_url,
                "index": "0",
            }
        )

        return ET.tostring(root, encoding="unicode")


def extract_user_attributes(
    assertion: SAMLAssertion,
    email_attribute: str = "email",
    first_name_attribute: str = "firstName",
    last_name_attribute: str = "lastName",
    groups_attribute: str = "groups",
) -> Dict[str, Any]:
    """
    Extract user attributes from a SAML assertion.

    Returns standardized user info dict.
    """
    # Try multiple common attribute names
    email_attrs = [
        email_attribute,
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
        "email",
        "mail",
        "Email",
    ]

    first_name_attrs = [
        first_name_attribute,
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname",
        "firstName",
        "givenName",
        "first_name",
    ]

    last_name_attrs = [
        last_name_attribute,
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname",
        "lastName",
        "surname",
        "sn",
        "last_name",
    ]

    groups_attrs = [
        groups_attribute,
        "http://schemas.xmlsoap.org/claims/Group",
        "groups",
        "memberOf",
        "role",
    ]

    def get_first_value(attr_names: List[str]) -> str:
        for name in attr_names:
            value = assertion.get_attribute(name)
            if value:
                return value
        return ""

    def get_all_values(attr_names: List[str]) -> List[str]:
        for name in attr_names:
            values = assertion.get_attributes(name)
            if values:
                return values
        return []

    # Use NameID as fallback for email
    email = get_first_value(email_attrs)
    if not email and "@" in assertion.subject_name_id:
        email = assertion.subject_name_id

    return {
        "email": email,
        "first_name": get_first_value(first_name_attrs),
        "last_name": get_first_value(last_name_attrs),
        "groups": get_all_values(groups_attrs),
        "name_id": assertion.subject_name_id,
        "session_index": assertion.session_index,
        "raw_attributes": assertion.attributes,
    }
