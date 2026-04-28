import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FirebaseService:
    """Optional Firebase integration layer.

    If firebase-admin or credentials are not available, methods gracefully
    return and callers can fall back to local storage.
    
    Fixes for incompatibilities:
    - Proper timestamp handling (client-generated fallback)
    - Query result limiting (max 1000 items per query)
    - Batch deletion support
    - Comprehensive error handling
    """

    def __init__(self) -> None:
        self.enabled = False
        self._auth = None
        self._firestore = None
        self._server_timestamp = None
        self._firebase_admin = None
        self._credentials = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            import firebase_admin
            from firebase_admin import auth, credentials, firestore
            self._firebase_admin = firebase_admin
        except ImportError as exc:
            logger.info("firebase-admin not installed: %s", exc)
            return
        except Exception as exc:
            logger.warning("Unexpected error importing firebase-admin: %s", exc)
            return

        try:
            # Check if app is already initialized
            if firebase_admin._apps:
                self.enabled = True
                self._auth = auth
                try:
                    self._firestore = firestore.client()
                    self._server_timestamp = firestore.SERVER_TIMESTAMP
                except Exception as e:
                    logger.warning("Failed to get Firestore client from existing app: %s", e)
                    self.enabled = False
                return
        except Exception as exc:
            logger.debug("Error checking existing Firebase apps: %s", exc)

        # Try to initialize Firebase with credentials
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
        cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

        cred = None
        
        # Try file-based credentials first
        if cred_path:
            if os.path.exists(cred_path):
                try:
                    cred = credentials.Certificate(cred_path)
                    logger.info("Loaded Firebase credentials from file: %s", cred_path)
                except Exception as exc:
                    logger.warning("Failed to load credentials from %s: %s", cred_path, exc)
            else:
                logger.warning("Firebase credentials file not found at: %s", cred_path)
        
        # Try JSON-based credentials if file didn't work
        if not cred and cred_json:
            try:
                cred_data = json.loads(cred_json)
                cred = credentials.Certificate(cred_data)
                logger.info("Loaded Firebase credentials from environment JSON")
            except json.JSONDecodeError as exc:
                logger.warning("Invalid JSON in FIREBASE_CREDENTIALS_JSON: %s", exc)
            except Exception as exc:
                logger.warning("Failed to create credentials from JSON: %s", exc)

        if not cred:
            logger.info(
                "Firebase credentials not configured. "
                "Set FIREBASE_CREDENTIALS_PATH or FIREBASE_CREDENTIALS_JSON to enable Firebase."
            )
            return

        try:
            firebase_admin.initialize_app(cred)
            self.enabled = True
            self._auth = auth
            self._firestore = firestore.client()
            self._server_timestamp = firestore.SERVER_TIMESTAMP
            logger.info("✓ Firebase initialized successfully")
        except Exception as exc:
            logger.warning("Failed to initialize Firebase: %s", exc)
            self.enabled = False

    def verify_bearer_token(self, auth_header: Optional[str]) -> Optional[Dict[str, Any]]:
        """Verify Firebase ID token from Authorization header.
        
        Expected format: "Bearer <token>"
        
        Returns:
            User info dict with uid/email/name/picture, or None if invalid/not enabled
        """
        if not self.enabled or not auth_header:
            return None
        
        if not isinstance(auth_header, str):
            logger.warning("Authorization header is not a string")
            return None
        
        if not auth_header.startswith("Bearer "):
            logger.debug("Authorization header does not start with 'Bearer '")
            return None

        token = auth_header.split(" ", 1)[1].strip() if " " in auth_header else ""
        if not token:
            logger.warning("Empty token in Authorization header")
            return None

        try:
            decoded = self._auth.verify_id_token(token)
            user_info = {
                "uid": decoded.get("uid"),
                "email": decoded.get("email"),
                "name": decoded.get("name"),
                "picture": decoded.get("picture"),
            }
            logger.debug("Token verified for user: %s", user_info.get("uid"))
            return user_info
            
        except Exception as exc:
            logger.warning("Token verification failed: %s", exc)
            return None

    def upsert_user_profile(self, user: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> bool:
        """Create or update a user profile.
        
        Args:
            user: User data (uid, email, name, picture)
            extra: Additional data to merge
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        uid = user.get("uid")
        if not uid:
            logger.warning("Cannot upsert user profile: uid is missing")
            return False

        try:
            payload = {
                "uid": uid,
                "email": user.get("email"),
                "display_name": user.get("name"),
                "photo_url": user.get("picture"),
                "updated_at": self._server_timestamp or datetime.utcnow().isoformat(),
            }
            
            if extra:
                payload.update(extra)

            self._firestore.collection("users").document(uid).set(payload, merge=True)
            return True
            
        except Exception as exc:
            logger.error("Error upserting user profile for %s: %s", uid, exc)
            return False

    def save_detection_log(self, log_entry: Dict[str, Any], user: Optional[Dict[str, Any]]) -> bool:
        return bool(self.save_forensic_log(log_entry, user))

    def save_forensic_log(
        self,
        log_entry: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Save a forensic log entry to Firestore.
        
        Args:
            log_entry: Log entry data to save
            user: Optional user info (includes uid, email)
            
        Returns:
            The saved payload with id, or None if Firebase is disabled or error occurs
        """
        if not self.enabled:
            return None

        try:
            payload = dict(log_entry)
            
            # Ensure timestamp is set
            if "timestamp" not in payload:
                payload["timestamp"] = datetime.utcnow().isoformat()
            
            payload.setdefault("source_type", "upload")

            # User info
            if user and user.get("uid"):
                payload["user_id"] = user.get("uid")
                payload["user_email"] = user.get("email")
            
            # Created timestamp - use server timestamp if available, otherwise client time
            if self._server_timestamp is not None:
                payload["created_at"] = self._server_timestamp
            else:
                payload["created_at"] = datetime.utcnow().isoformat()
            
            # Save to Firestore
            doc_ref = self._firestore.collection("forensic_logs").document()
            payload["id"] = doc_ref.id
            doc_ref.set(payload)
            
            return payload
            
        except Exception as exc:
            logger.error("Error saving forensic log to Firebase: %s", exc)
            return None

    def _normalize_log_doc(self, doc: Any) -> Dict[str, Any]:
        item = doc.to_dict() or {}
        item["id"] = item.get("id") or doc.id

        created_at = item.get("created_at")
        if hasattr(created_at, "isoformat"):
            item["created_at"] = created_at.isoformat()

        timestamp = item.get("timestamp")
        if hasattr(timestamp, "isoformat"):
            item["timestamp"] = timestamp.isoformat()
        elif item.get("timestamp") is None:
            item["timestamp"] = datetime.utcnow().isoformat()
        return item

    def get_forensic_logs(
        self,
        page: int = 1,
        page_size: int = 50,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source_type: Optional[str] = None,
        user: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get forensic logs with pagination and filtering.
        
        Note: Firestore has a 1000-document limit per query. 
        This method implements proper pagination and error handling.
        """
        if not self.enabled:
            return {"items": [], "total": 0, "page": page, "page_size": page_size}

        try:
            from firebase_admin import firestore as fb_firestore
        except ImportError:
            logger.warning("firebase_admin.firestore not available")
            return {"items": [], "total": 0, "page": page, "page_size": page_size}

        try:
            # Build base query
            base_query = self._firestore.collection("forensic_logs")

            # Apply filters (Firestore requires ordered application)
            if user and user.get("uid"):
                base_query = base_query.where("user_id", "==", user.get("uid"))
            
            if source_type:
                base_query = base_query.where("source_type", "==", source_type)
            
            # Date range filters (note: timestamp must be ISO format string)
            if start_date:
                base_query = base_query.where("timestamp", ">=", start_date)
            
            if end_date:
                base_query = base_query.where("timestamp", "<=", end_date)

            try:
                # Count total (with 1000 doc limit warning)
                total = 0
                for _ in base_query.limit(1001).stream():
                    total += 1
                
                if total >= 1000:
                    logger.warning(
                        "Forensic logs query reached Firestore 1000-doc limit. "
                        "Consider adding more filters or archiving old logs."
                    )
            except Exception as e:
                logger.warning("Could not count total logs: %s", e)
                total = 0

            # Calculate pagination
            offset = max(0, (page - 1) * page_size)
            
            # Validate page_size (Firestore max 1000 per query)
            if page_size > 1000:
                logger.warning("page_size %d exceeds Firestore limit of 1000, capping to 1000", page_size)
                page_size = 1000

            # Build paginated query
            page_query = (
                base_query
                .order_by("timestamp", direction=fb_firestore.Query.DESCENDING)
                .offset(offset)
                .limit(page_size)
            )

            items = [self._normalize_log_doc(doc) for doc in page_query.stream()]
            
            return {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
            }
            
        except Exception as exc:
            logger.error("Error retrieving forensic logs: %s", exc)
            return {"items": [], "total": 0, "page": page, "page_size": page_size}

    def get_detection_logs(self, limit: int = 50, user: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        response = self.get_forensic_logs(
            page=1,
            page_size=limit,
            source_type="upload",
            user=user,
        )
        return response.get("items", [])

    def delete_forensic_log(self, log_id: str, user: Optional[Dict[str, Any]] = None) -> bool:
        if not self.enabled or not log_id:
            return False

        doc_ref = self._firestore.collection("forensic_logs").document(log_id)
        doc = doc_ref.get()
        if not doc.exists:
            return False

        payload = doc.to_dict() or {}
        if user and user.get("uid") and payload.get("user_id") != user.get("uid"):
            return False

        doc_ref.delete()
        return True

    def clear_forensic_logs(
        self,
        user: Optional[Dict[str, Any]] = None,
        source_type: Optional[str] = None,
    ) -> int:
        """Delete multiple forensic log entries (with batch operation support).
        
        Returns:
            Number of documents deleted
        """
        if not self.enabled:
            return 0

        try:
            # Build query
            query = self._firestore.collection("forensic_logs")

            if user and user.get("uid"):
                query = query.where("user_id", "==", user.get("uid"))
            if source_type:
                query = query.where("source_type", "==", source_type)

            # Get docs to delete (with 1000-doc limit)
            docs = query.limit(1000).stream()
            doc_list = list(docs)
            
            if not doc_list:
                return 0

            # Batch delete (Firestore batches support max 500 writes)
            batch_size = 500
            deleted = 0
            
            for i in range(0, len(doc_list), batch_size):
                batch = self._firestore.batch()
                batch_docs = doc_list[i : i + batch_size]
                
                for doc in batch_docs:
                    batch.delete(doc.reference)
                
                try:
                    batch.commit()
                    deleted += len(batch_docs)
                except Exception as e:
                    logger.error("Error committing batch delete: %s", e)
            
            logger.info("Deleted %d forensic logs", deleted)
            return deleted
            
        except Exception as exc:
            logger.error("Error clearing forensic logs: %s", exc)
            return 0

    def get_user_profile(self, uid: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or not uid:
            return None

        doc = self._firestore.collection("users").document(uid).get()
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        data["uid"] = uid
        return data
