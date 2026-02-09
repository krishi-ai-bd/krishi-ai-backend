from fastapi import APIRouter

# This file is deprecated - using new document service instead
# See: app/services/documents/document_route.py

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])

# Old product/service routes have been replaced by the new PDF-based
# document management system with RAG capabilities 

@router.post("/products", response_model=OperationResponse)
async def add_product(product: ProductKnowledge):
    """Add a new product to the knowledge base"""
    return product_knowledge_manager.add_product(product)


@router.put("/products/{product_id}", response_model=OperationResponse)
async def update_product(product_id: str, product: ProductKnowledge):
    """Update an existing product"""
    return product_knowledge_manager.update_product(product_id, product)


@router.delete("/products/{product_id}", response_model=OperationResponse)
async def delete_product(product_id: str):
    """Delete a product from the knowledge base"""
    return product_knowledge_manager.delete_product(product_id)


@router.get("/products/search", response_model=List[SearchResult])
async def search_products(
    query: str = Query(..., description="Search query"),
    n_results: int = Query(5, ge=1, le=50, description="Number of results"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Search products using semantic similarity"""
    filters = {"category": category} if category else None
    return product_knowledge_manager.search_products(query, n_results, filters)


@router.get("/products/{product_id}")
async def get_product(product_id: str):
    """Get a specific product by ID"""
    product = product_knowledge_manager.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.get("/products", response_model=List[Dict])
async def get_all_products(limit: int = Query(100, ge=1, le=1000)):
    """Get all products"""
    return product_knowledge_manager.get_all_products(limit)


# SERVICE ROUTES 

@router.post("/services", response_model=OperationResponse)
async def add_service(service: ServiceKnowledge):
    """Add a new service to the knowledge base"""
    return service_knowledge_manager.add_service(service)


@router.put("/services/{service_id}", response_model=OperationResponse)
async def update_service(service_id: str, service: ServiceKnowledge):
    """Update an existing service"""
    return service_knowledge_manager.update_service(service_id, service)


@router.delete("/services/{service_id}", response_model=OperationResponse)
async def delete_service(service_id: str):
    """Delete a service from the knowledge base"""
    return service_knowledge_manager.delete_service(service_id)


@router.get("/services/search", response_model=List[SearchResult])
async def search_services(
    query: str = Query(..., description="Search query"),
    n_results: int = Query(5, ge=1, le=50, description="Number of results"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Search services using semantic similarity"""
    filters = {"category": category} if category else None
    return service_knowledge_manager.search_services(query, n_results, filters)


@router.get("/services/{service_id}")
async def get_service(service_id: str):
    """Get a specific service by ID"""
    service = service_knowledge_manager.get_service_by_id(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    return service


@router.get("/services", response_model=List[Dict])
async def get_all_services(limit: int = Query(100, ge=1, le=1000)):
    """Get all services"""
    return service_knowledge_manager.get_all_services(limit)


# CONSULTATION ROUTES

@router.post("/consultations", response_model=OperationResponse)
async def add_consultation(consultation: ConsultationKnowledge):
    """Add a new consultation to the knowledge base"""
    return consultation_knowledge_manager.add_consultation(consultation)


@router.put("/consultations/{consultation_id}", response_model=OperationResponse)
async def update_consultation(consultation_id: str, consultation: ConsultationKnowledge):
    """Update an existing consultation"""
    return consultation_knowledge_manager.update_consultation(consultation_id, consultation)


@router.delete("/consultations/{consultation_id}", response_model=OperationResponse)
async def delete_consultation(consultation_id: str):
    """Delete a consultation from the knowledge base"""
    return consultation_knowledge_manager.delete_consultation(consultation_id)


@router.get("/consultations/search", response_model=List[SearchResult])
async def search_consultations(
    query: str = Query(..., description="Search query"),
    n_results: int = Query(5, ge=1, le=50, description="Number of results"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Search consultations using semantic similarity"""
    filters = {"category": category} if category else None
    return consultation_knowledge_manager.search_consultations(query, n_results, filters)


@router.get("/consultations/{consultation_id}")
async def get_consultation(consultation_id: str):
    """Get a specific consultation by ID"""
    consultation = consultation_knowledge_manager.get_consultation_by_id(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return consultation


@router.get("/consultations", response_model=List[Dict])
async def get_all_consultations(limit: int = Query(100, ge=1, le=1000)):
    """Get all consultations"""
    return consultation_knowledge_manager.get_all_consultations(limit)


# SPECIALIST ROUTES

@router.post("/specialists", response_model=OperationResponse)
async def add_specialist(specialist: SpecialistKnowledge):
    """Add a new specialist to the knowledge base"""
    return specialist_knowledge_manager.add_specialist(specialist)


@router.put("/specialists/{specialist_id}", response_model=OperationResponse)
async def update_specialist(specialist_id: str, specialist: SpecialistKnowledge):
    """Update an existing specialist"""
    return specialist_knowledge_manager.update_specialist(specialist_id, specialist)


@router.delete("/specialists/{specialist_id}", response_model=OperationResponse)
async def delete_specialist(specialist_id: str):
    """Delete a specialist from the knowledge base"""
    return specialist_knowledge_manager.delete_specialist(specialist_id)


@router.get("/specialists/search", response_model=List[SearchResult])
async def search_specialists(
    query: str = Query(..., description="Search query"),
    n_results: int = Query(5, ge=1, le=50, description="Number of results"),
    category: Optional[str] = Query(None, description="Filter by category"),
    min_rating: Optional[float] = Query(None, ge=0, le=5, description="Minimum rating")
):
    """Search specialists using semantic similarity"""
    filters = {}
    if category:
        filters["category"] = category
    if min_rating is not None:
        filters["rating"] = {"$gte": min_rating}
    
    return specialist_knowledge_manager.search_specialists(
        query, 
        n_results, 
        filters if filters else None
    )


@router.get("/specialists/{specialist_id}")
async def get_specialist(specialist_id: str):
    """Get a specific specialist by ID"""
    specialist = specialist_knowledge_manager.get_specialist_by_id(specialist_id)
    if not specialist:
        raise HTTPException(status_code=404, detail="Specialist not found")
    return specialist


@router.get("/specialists", response_model=List[Dict])
async def get_all_specialists(limit: int = Query(100, ge=1, le=1000)):
    """Get all specialists"""
    return specialist_knowledge_manager.get_all_specialists(limit)
