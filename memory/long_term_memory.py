"""
long_term_memory.py - main qdrant class for storage of ltm

Memoir+ a persona extension for Text Gen Web UI. 
MIT License

Copyright (c) 2024 brucepro

additional info here https://python-client.qdrant.tech/

"""

import os
import random
from datetime import datetime, timedelta
from modules.logging_colors import logger
from qdrant_client import models, QdrantClient
from qdrant_client.http.models import PointStruct
from html import escape
from sentence_transformers import SentenceTransformer


class LTM():
    def __init__(
            self,
            collection,
            ltm_limit,
            verbose=False,
            embedder='all-mpnet-base-v2',
            address='localhost',
            port=6333,
            device='cuda'
    ):

        self.verbose = verbose
        if self.verbose:
            logger.debug(f'[Memoir] [{self.__class__.__name__}] Verbose mode enabled.')
        self.collection = collection
        self.ltm_limit = ltm_limit
        if self.verbose:
            logger.debug(f"[Memoir] [{self.__class__.__name__}] LIMIT: {str(ltm_limit)}")
        self.address = address
        self.port = port
        if self.verbose:
            logger.debug(f'[Memoir] [{self.__class__.__name__}] [QDRANT] addr:{self.address}, port:{self.port}')

        self.embedder = embedder
        self.device = device
        # Check if specified embedder is already downloaded to load it from file
        filePath = os.path.dirname(os.path.realpath(__file__))
        basePath = os.path.abspath(os.path.join(filePath, os.pardir))
        embedderSavePath = os.path.join(
            basePath + os.sep,
            "storage" + os.sep,
            "models" + os.sep,
            self.embedder
        )
        embedderSaveFile = os.path.join(
            embedderSavePath + os.sep,
            "model.safetensors"
        )
        if os.path.exists(embedderSavePath) and os.path.isfile(embedderSaveFile):
            if self.verbose:
                logger.debug(f'[Memoir] [{self.__class__.__name__}] embedder {self.embedder} found in {embedderSavePath}, will try to load from disk.')
            try:
                self.encoder = SentenceTransformer(embedderSavePath)  # , device=self.device)
            except Exception as e:
                logger.error(f'[Memoir] [{self.__class__.__name__}] There was an error while loading {self.embedder}, please check.')
        else:
            if self.verbose:
                logger.warning(f'[Memoir] [{self.__class__.__name__}] embedder {self.embedder} NOT FOUND in {embedderSavePath}, will be downloaded and saved to disk.')
            self.encoder = SentenceTransformer(self.embedder)  # , device=self.device)
            self.encoder.save(embedderSavePath)

        self.qdrant = QdrantClient(self.address, port=self.port)
        self.create_vector_db_if_missing()

    def create_vector_db_if_missing(self):
        if self.qdrant.collection_exists(collection_name=self.collection) == True:
            if self.verbose:
                logger.debug(f'[Memoir] [{self.__class__.__name__}] Collection {self.collection} found.')
                vectors_count = self.qdrant.get_collection(self.collection).vectors_count
                points_count = self.qdrant.get_collection(self.collection).points_count
                logger.debug(f'[Memoir] [{self.__class__.__name__}] self.collection: {self.collection} already exists with {vectors_count} vectors and {points_count} points.')
        else:
            try:
                self.qdrant.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=self.encoder.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE
                    )
                )
                if self.verbose:
                    logger.debug(f'[Memoir] [{self.__class__.__name__}] Created self.collection: {self.collection}')
            except Exception as e:
                if self.verbose:
                    logger.debug(f'[Memoir] [{self.__class__.__name__}] There was an error while creating self.collection: {self.collection}: {e}')

    def delete_vector_db(self):
        try:
            self.qdrant.delete_collection(
                collection_name=self.collection,
            )
            if self.verbose:
                logger.debug(f'[Memoir] [{self.__class__.__name__}] Deleted self.collection: {self.collection}')
        except Exception as e:
            if self.verbose:
                logger.debug(f'[Memoir] [{self.__class__.__name__}] self.collection: {self.collection} does not exist, not deleting: {e}')

    def store(self, doc_to_upsert):
        operation_info = self.qdrant.upsert(
            collection_name=self.collection,
            wait=True,
            points=self.get_embedding_vector(doc_to_upsert),
        )
        if self.verbose:
            print(operation_info)

    def get_embedding_vector(self, doc):
        data = doc['comment'] + doc['people']
        self.vector = self.encoder.encode(data).tolist()
        self.next_id = random.randint(0, 1e10)
        points = [
            PointStruct(id=self.next_id,
                        vector=self.vector,
                        payload=doc),
        ]
        return points

    def recall(self, query):
        self.query_vector = self.encoder.encode(query).tolist()
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=self.query_vector,
            limit=self.ltm_limit + 1
        )
        return self.format_results_from_qdrant(results)

    def delete(self, comment_id):
        self.qdrant.delete_points(self.collection, [comment_id])
        self.logger.debug(f"Deleted comment with ID: {comment_id}")

    def format_results_from_qdrant(self, results):
        formated_results = []
        seen_comments = set()
        result_count = 0
        for result in results[1:]:
            comment = result.payload['comment']
            if comment not in seen_comments:
                seen_comments.add(comment)
                # formated_results.append("(" + result.payload['datetime'] + "'Memory':" + result.payload['comment'] + ", 'Emotions:" + result.payload['emotions'] + ", 'People':" + result.payload['people'] + ")")
                # formated_results.append("(" + result.payload['datetime'] + "Memory:" + escape(result.payload['comment']) + ",Emotions:" + escape(result.payload['emotions']) + ",People:" + escape(result.payload['people']) + ")")
                datetime_obj = datetime.strptime(result.payload['datetime'], "%Y-%m-%dT%H:%M:%S.%f")
                date_str = datetime_obj.strftime("%Y-%m-%d")
                formated_results.append(result.payload['comment'] + ": on " + str(date_str))
            else:
                if self.verbose:
                    logger.debug(f"[Memoir] [{self.__class__.__name__}] Not adding {comment}")
            result_count += 1
        return formated_results

    def get_last_summaries(self, range):
        # Retrieve all vectors
        query_vector = self.encoder.encode("").tolist()
        all_vectors = self.qdrant.search(collection_name=self.collection, query_vector=query_vector, limit=99999999999 + 1)
        formated_results = []
        seen_comments = set()
        result_count = 0
        for result in all_vectors:
            comment = result.payload['comment']
            if comment not in seen_comments:
                seen_comments.add(comment)
                datetime_obj = datetime.strptime(result.payload['datetime'], "%Y-%m-%dT%H:%M:%S.%f")
                date_str = datetime_obj.strftime("%Y-%m-%d")
                now = datetime.now()
                if now - datetime_obj <= timedelta(hours=range):
                    if self.verbose:
                        logger.debug(f"[Memoir] [{self.__class__.__name__}] Adding memory to stm_context")
                    formated_results.append(result.payload['comment'] + ": on " + str(date_str))
            else:
                if self.verbose:
                    logger.info(f"[Memoir] [{self.__class__.__name__}] Not adding {comment}")
            result_count += 1
        return formated_results

    def __repr__(self):
        return f"address: {self.address}, collection: {self.collection}"

    def __len__(self):
        return self.qdrant.get_collection(self.collection).vectors_count
